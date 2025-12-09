import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MCPFilter:
    def __init__(self, particles, sigma, alpha, seed=20):
        # 粒子フィルタで用いる粒子数 N
        self.particles = particles
        # 観測ノイズの分散（正確には sigma = 観測ノイズの分散）
        # 観測モデル:  y_t = x_t + e_t,  e_t ~ N(0, sigma)
        self.sigma = sigma
        # 状態遷移ノイズの分散係数
        # 状態モデル:  x_t = x_{t-1} + v_t,  v_t ~ N(0, alpha * sigma)
        self.alpha = alpha
        # 乱数シード（再現性確保用）
        self.seed = seed

    def k_val(self, w_cumsum, idx, u):
        """
        系統的リサンプリング(systematic resampling)で
        一様乱数 u に対応する粒子インデックス k を返す関数。

        w_cumsum: 正規化済み重みの累積和  (長さ N)
        idx     : 粒子のインデックス配列 np.arange(N)
        u       : [0,1] 上のサンプル点

        制度的には「CDF が u を初めて超える点のインデックス」に対応する。
        """
        if not (w_cumsum < u).any():
            # u が全ての累積和よりも小さい場合 → 先頭の粒子を選択
            return 0
        else:
            # w_cumsum < u を満たす最大のインデックス + 1 が
            # CDF が u を超える最初の位置を表す
            return (idx[w_cumsum < u]).max() + 1

    def mc_sampling(self, weights):
        """
        系統的リサンプリングの本体。

        weights: 各粒子の正規化重み w_t^{(i)} (和が 1)

        手順:
          1. 区間 [0, 1/N] からの一様乱数 u0 を 1 つ生成
          2. U_j = (j / N) + u0, j = 0,...,N-1 を作る
             → [0,1] を N 個の均等な区間に区切り、その中央付近をサンプルするイメージ
          3. 各 U_j に対して、累積分布 w_cumsum に基づき粒子を選択

        これにより、重み分布に比例したサンプリングを
        低分散で実現できる。
        """
        idx = np.arange(self.particles)
        # [0, 1/N] の一様乱数
        u0 = np.random.uniform(0, 1 / self.particles)
        # 系統的に並んだサンプル点 U
        U = np.arange(self.particles) / self.particles + u0
        # 重みの累積和（CDF）
        w_cumsum = np.cumsum(weights)
        # 各 U_j に対して対応する粒子インデックスを求める
        k_list = np.array([self.k_val(w_cumsum, idx, val) for val in U])
        return k_list

    def filtering(self, y):
        """
        粒子フィルタによる 1 次元状態空間モデルのフィルタリング。

        モデル想定:
          状態方程式: x_{t+1} = x_t + v_t,  v_t ~ N(0, alpha * sigma)
          観測方程式: y_t     = x_t + e_t,  e_t ~ N(0, sigma)

        目的:
          - 各時刻 t での事後分布 p(x_t | y_{1:t}) を粒子集合で近似
          - 対数尤度 log p(y_{1:T}) の近似値を計算
        """
        # 乱数シードの固定（再現性のため）
        np.random.seed(self.seed)

        T = len(y)  # 観測系列の長さ T

        # x[t, i]          : 時刻 t の i 番目粒子の状態
        # x_resample[t, i] : リサンプリング後の状態（次ステップの遷移元）
        x = np.zeros((T + 1, self.particles))
        x_resample = np.zeros((T + 1, self.particles))

        # 初期状態 x_0 を N(0,1) からサンプリング
        # → p(x_0) の事前分布を近似していると解釈できる
        x_init = np.random.normal(0, 1, size=self.particles)
        x_resample[0] = x_init
        x[0] = x_init

        # w[t, i]               : t 時刻の i 番目粒子の「非正規化重み」
        # normalized_weight[t,i]: 正規化後の重要度重み
        w = np.zeros((T, self.particles))
        normalized_weight = np.zeros((T, self.particles))

        # l[t]: t 時刻の尤度項 log( 1/N * Σ_i w_t^{(i)} ) の内部 log Σ_i w_t^{(i)} に相当
        l = np.zeros(T)

        # 時刻 t = 0,...,T-1 に対して逐次的にフィルタリング
        for t in range(T):
            for i in range(self.particles):
                # 状態遷移: x_{t+1}^{(i)} = x_t^{(i)} + v_t^{(i)},  v_t^{(i)} ~ N(0, alpha * sigma)
                v = np.random.normal(0, np.sqrt(self.alpha * self.sigma))
                x[t + 1, i] = x_resample[t, i] + v

                # 観測尤度 p(y_t | x_{t+1}^{(i)})
                # 観測モデル: y_t = x_{t+1}^{(i)} + e_t, e_t ~ N(0, sigma)
                # よって
                #   p(y_t | x_{t+1}^{(i)}) =
                #     (1 / sqrt(2π sigma)) * exp(-(y_t - x_{t+1}^{(i)})^2 / (2 sigma))
                w[t, i] = np.exp(
                    -((y[t] - x[t + 1, i]) ** 2) / (2 * self.sigma)
                ) / np.sqrt(2 * np.pi * self.sigma)

            # 式(86)に関する計算（重要度重みの正規化）
            # normalized_weight[t, i] ∝ p(y_t | x_{t+1}^{(i)}) で
            # Σ_i normalized_weight[t, i] = 1 となるようにスケーリング
            normalized_weight[t] = w[t] / np.sum(w[t])

            # 重要度重みに基づくリサンプリング（退化の防止）
            # → resample 後の粒子集合は、事後分布 p(x_{t+1} | y_{1:t}) の近似とみなせる
            k = self.mc_sampling(normalized_weight[t])
            x_resample[t + 1] = x[t + 1, k]

            # 時刻 t における (非正規化) 尤度 Σ_i w_t^{(i)} の対数を記録
            #   log p(y_t | y_{1:t-1}) ≈ log( (1/N) Σ_i w_t^{(i)} )
            # となるので、後で全時刻分を足し合わせて系列全体の対数尤度を近似する
            l[t] = np.log(np.sum(w[t]))

        # 系列全体の対数尤度の近似
        #   log p(y_{1:T}) = Σ_t log p(y_t | y_{1:t-1})
        # ≈ Σ_t log( (1/N) Σ_i w_t^{(i)} )
        # = Σ_t [ log Σ_i w_t^{(i)} - log N ]
        log_likelihood = np.sum(l) - T * np.log(self.particles)

        # フィルタ分布の期待値 E[x_t | y_{1:t}] の近似
        # normalized_weight[t, :] を重みとした粒子の重み付き平均を取っている。
        # 行列積 normalized_weight * x[1:].T の対角成分が各時刻の期待値に対応。
        filtered_value = np.diag(np.dot(normalized_weight, x[1:].T))

        # x: 各時刻・各粒子の軌跡
        # filtered_value: 各時刻のフィルタ平均
        # log_likelihood: 観測系列全体の対数尤度近似値
        return x, filtered_value, log_likelihood


def main():
    # 航空旅客数データ(AirPassengers)の読み込み
    # df: 各月の旅客数（時系列データ）
    content = pd.read_csv("./data/AirPassengers.csv")
    df = content["#Passengers"]

    # 粒子数とノイズのハイパーパラメータの設定
    particles = 200  # 粒子数 N
    sigma = 15  # 観測ノイズの分散
    alpha = 5  # 状態ノイズの分散係数（alpha * sigma）

    # 粒子フィルタモデルの生成とフィッティング
    model = MCPFilter(particles, sigma, alpha)
    X, filtered_value, log_likelihood = model.filtering(df)

    # 観測値・フィルタ平均・粒子の軌跡の可視化
    plt.figure(figsize=(12, 6))
    # 黒線: 観測データ y_t
    plt.plot(df.values, ".-k", label="data")
    # 緑線: フィルタ平均 E[x_t | y_{1:t}]
    plt.plot(filtered_value, ".--g", label="filtered")
    # 赤点: 各時刻における粒子の位置（状態のサンプル）
    for t in range(len(df)):
        plt.scatter(np.ones(particles) * t, X[t], color="r", s=1, alpha=0.6)

    # タイトルにはパラメータと対数尤度を表示
    plt.title(f"sigma^2={sigma}, alpha^2={alpha}, log likelihood={log_likelihood :.3f}")
    plt.legend()
    plt.xlabel("input data length")
    plt.ylabel("input and generated data value")
    plt.show()


if __name__ == "__main__":
    main()
