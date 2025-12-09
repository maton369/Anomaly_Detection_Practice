import numpy as np
import matplotlib.pyplot as plt


class GaussianMixture:
    def __init__(self, total_component):
        # 混合ガウス分布の成分数 K
        self.total_component = total_component

    # 各成分の多変量ガウス密度 N(x | μ_k, Σ_k) をまとめて計算する関数
    # ガウス混合モデルの E ステップ・M ステップの両方で利用する
    def G_M_calcultaion(self, inputs):
        # 共分散行列 Σ_k の逆行列 Σ_k^{-1} を成分ごとに計算
        # current_covs の shape は (D, D, K) なので，
        # 転置して (K, D, D) にした上で np.linalg.inv を適用し，
        # 再度 (D, D, K) に戻している
        accuracy = np.linalg.inv(self.current_covs.T).T

        # 各データ点 x_n から各成分の平均 μ_k を引いた差 (x_n - μ_k)
        # inputs: (N, D)
        # inputs[:, :, None]: (N, D, 1)
        # current_mean: (D, K) → ブロードキャストで (1, D, K)
        # changed: (N, D, K)
        changed = inputs[:, :, None] - self.current_mean

        # exp_calculation は指数部の二次形式
        # (x_n - μ_k)^T Σ_k^{-1} (x_n - μ_k) を全 n, k について計算している
        # einsum によって，内積部分 (Σ_k^{-1} (x_n - μ_k)) を一括で計算
        exp_calculation = np.sum(
            np.einsum("nik,ijk->njk", changed, accuracy) * changed, axis=1
        )  # shape: (N, K)

        # 多変量正規分布の密度
        # N(x | μ_k, Σ_k)
        # = (2π)^{-D/2} |Σ_k|^{-1/2} exp( -1/2 (x-μ_k)^T Σ_k^{-1} (x-μ_k) )
        return np.exp(-0.5 * exp_calculation) / np.sqrt(
            np.linalg.det(self.current_covs.T).T * (2 * np.pi) ** self.dimention_size
        )

    # EM アルゴリズムにより混合ガウスモデルを学習するメイン関数
    def fit(self, inputs, maximum_iteration=10):
        # データ次元 D
        self.dimention_size = inputs.shape[1]

        # 初期混合係数 π_k（すべて一様に 1/K とする）
        self.current_weight = np.ones(self.total_component) / self.total_component

        # 平均ベクトル μ_k の初期値
        # データ全体の min〜max の一様分布で乱数初期化
        # shape: (D, K)
        self.current_mean = np.random.uniform(
            inputs.min(), inputs.max(), (self.dimention_size, self.total_component)
        )

        # 共分散行列 Σ_k の初期値
        # 各成分とも 10 I_D から開始する
        # shape: (D, D, K)
        self.current_covs = np.repeat(
            10 * np.eye(self.dimention_size), self.total_component
        ).reshape(self.dimention_size, self.dimention_size, self.total_component)

        # EM アルゴリズムの反復
        for i in range(maximum_iteration):
            # 収束判定のために現在パラメータをフラット化して保存
            current_params = np.hstack(
                (
                    self.current_weight.ravel(),
                    self.current_mean.ravel(),
                    self.current_covs.ravel(),
                )
            )

            # E-step: 事後確率（responsibility）γ_{nk} を計算
            current_resps = self.belife(inputs)

            # M-step: γ_{nk} を固定してパラメータ (π_k, μ_k, Σ_k) を更新
            self.make_maximum(inputs, current_resps)

            # パラメータが十分小さくしか変化しなくなったら収束とみなす
            if np.allclose(
                current_params,
                np.hstack(
                    (
                        self.current_weight.ravel(),
                        self.current_mean.ravel(),
                        self.current_covs.ravel(),
                    )
                ),
            ):
                break
        else:
            # 最大反復まで回しても収束しなかった場合
            print("parameters have not converged")

    # E-step
    # γ_{nk} = p(z_k = 1 | x_n) を計算する
    def belife(self, inputs):
        # まず事前分布 π_k と尤度 N(x_n | μ_k, Σ_k) の積
        # π_k N(x_n | μ_k, Σ_k) = joint p(x_n, z_k)
        current_resps = self.current_weight * self.G_M_calcultaion(inputs)

        # 各 n について成分方向に正規化して事後確率 γ_{nk} に変換
        # γ_{nk} = π_k N(x_n | μ_k, Σ_k) / Σ_j π_j N(x_n | μ_j, Σ_j)
        current_resps /= current_resps.sum(axis=-1, keepdims=True)
        return current_resps

    # M-step
    # γ_{nk} を固定したもとで，対数尤度を最大化するように
    # π_k, μ_k, Σ_k を更新する
    def make_maximum(self, inputs, current_resps):
        # 各成分 k に割り当てられた「有効サンプル数」N_k
        # N_k = Σ_n γ_{nk}
        Nk = np.sum(current_resps, axis=0)

        # 混合係数 π_k の更新
        # π_k = N_k / N
        self.current_weight = Nk / len(inputs)

        # 平均 μ_k の更新
        # μ_k = (1 / N_k) Σ_n γ_{nk} x_n
        # inputs.T: (D, N), current_resps: (N, K)
        # → (D, K)
        self.current_mean = np.dot(inputs.T, current_resps) / Nk

        # 共分散 Σ_k の更新
        # Σ_k = (1 / N_k) Σ_n γ_{nk} (x_n - μ_k)(x_n - μ_k)^T
        # changed: (N, D, K) で (x_n - μ_k) を持つ
        changed = inputs[:, :, None] - self.current_mean

        # einsum で各成分の共分散を一括計算
        # changed * current_resps: γ_{nk} (x_n - μ_k) を掛けてから
        # (n 方向に和を取る) → (D, D, K)
        self.current_covs = (
            np.einsum(
                "nik,njk->ijk", changed, changed * np.expand_dims(current_resps, 1)
            )
            / Nk
        )

    # 式(85)の計算
    # 事後確率 p(z_k = 1 | x) が最大となる成分インデックスを返す
    # すなわち「最も尤もらしいガウス成分によるクラスタリング」
    def type_sort(self, inputs):
        # joint_prob = π_k N(x | μ_k, Σ_k) （正規化前の値）
        joint_prob = self.current_weight * self.G_M_calcultaion(inputs)
        # 各サンプルが最も高い joint_prob を持つ成分に割り当てられる
        return np.argmax(joint_prob, axis=1)

    # データ x に対して混合分布 p(x) = Σ_k π_k N(x | μ_k, Σ_k) を計算する
    def probability_prediction(self, inputs):
        G_M_calcultaion = self.current_weight * self.G_M_calcultaion(inputs)
        return np.sum(G_M_calcultaion, axis=-1)


def main():
    # 3 つの 2 次元ガウス分布からサンプルを生成
    # x1: 左上，x2: 右下，x3: 原点付近
    x1 = np.random.normal(size=(100, 2)) + np.array([-5, 5])
    x2 = np.random.normal(size=(100, 2)) + np.array([5, -5])
    x3 = np.random.normal(size=(100, 2))

    # これらを縦方向に結合して 300 サンプルのデータセットを作る
    inputs = np.vstack((x1, x2, x3))

    # 成分数 K=3 の混合ガウスモデルを構築し EM で学習
    GM_model = GaussianMixture(3)
    GM_model.fit(inputs, maximum_iteration=30)

    # 各サンプルがどの成分に最も属しやすいかをクラスタラベルとして取得
    target = GM_model.type_sort(inputs)

    # 可視化のためのグリッド状のテスト点を生成し，
    # そこに対する混合分布の確率密度 p(x) を評価
    data_test, target_test = np.meshgrid(
        np.linspace(-10, 10, 100), np.linspace(-10, 10, 100)
    )
    Data_test = np.array([data_test, target_test]).reshape(2, -1).transpose()
    probs = GM_model.probability_prediction(Data_test)
    Probs = probs.reshape(100, 100)

    # 学習データを「属する成分ごと」の色でプロット
    colors = ["red", "blue", "green"]
    plt.scatter(inputs[:, 0], inputs[:, 1], c=[colors[int(label)] for label in target])

    # 背景に混合分布 p(x) の等高線を描画
    # 等高線は「確率密度が同じ値をとる曲線」であり，
    # 各ガウス成分の楕円形の輪郭や混合としての形状が視覚的に分かる
    plt.contour(data_test, target_test, Probs)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()


if __name__ == "__main__":
    main()
