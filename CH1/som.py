import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


class SOM:
    def __init__(self, learning_rate):
        # SOM の学習率 η₀ を保持
        # 学習の途中では lr_decay により指数的に減衰させる
        self.learning_rate = learning_rate

    # 式(82)に対応：入力ベクトル s に対して BMU（Best Matching Unit）を探す
    # BMU とは「重みベクトル w_k とのユークリッド距離が最小のユニット」のこと
    def best_matching_unit(self, s):
        x_bmu = np.array([0, 0])  # BMU の格子座標 (i, j)
        # 初期の最小距離を非常に大きな値にしておく
        minimum_distance = np.iinfo(np.int_).max

        # SOM の全てのユニット (k, j) について重みベクトルとの距離を調べる
        for k in range(self.structure.shape[0]):
            for j in range(self.structure.shape[1]):
                # 各ユニットの重みベクトル w_kj ∈ R^m
                weight = self.structure[k, j, :].reshape(self.m, 1)
                # 距離 ||s − w_kj||² を計算
                distance = np.sum((weight - s) ** 2)
                # 最小距離を与えるユニットを BMU とする
                if distance < minimum_distance:
                    minimum_distance = distance
                    x_bmu = np.array([k, j])

        # BMU の重みベクトルを返す
        best_mu = self.structure[x_bmu[0], x_bmu[1], :].reshape(self.m, 1)
        return best_mu, x_bmu, minimum_distance

    # 近傍半径 r(t) の時間変化
    # 一般的な SOM では
    #   r(t) = r_0 exp(-t / τ)
    # と指数的に減少させることで、序盤は広い範囲を更新し、
    # 後半は BMU 近傍だけを微調整する
    def change_radius(self, initial_radius, i, t_cons):
        return initial_radius * np.exp(-i / t_cons)

    # 学習率 η(t) の時間変化
    #   η(t) = η_0 exp(-t / T)
    # とすることで、学習が進むにつれて更新量を小さくする
    def lr_decay(self, learning_rate, i, iters):
        return learning_rate * np.exp(-i / iters)

    # 近傍関数 h(d, r) の計算
    #   h = exp(-d / (2 r²))
    # ここで d は BMU との格子距離（ユークリッド距離の二乗）、
    # r は現在の近傍半径
    # この関数は「BMU に近いユニットほど大きく更新し、遠いほど小さくする」
    # というガウス型の影響度を表す
    def changes(self, distance, r_data):
        return np.exp(-distance / (2 * (r_data**2)))

    # SOM の学習本体
    # list_data: 入力ベクトル列 x_n
    # classes: 想定クラス数（ここではマップの大きさ決定に使っている）
    # iterations: 繰り返し回数
    def som_training(self, list_data, classes, iterations):
        # n: サンプル数, m: 特徴次元
        self.n, self.m = list_data.shape

        # マップの格子サイズを (2*classes) x (2*classes) に設定
        # 理論的には「トポロジ保存写像」を学習するので、
        # 2 次元格子上のノードが高次元空間の構造を近似的に表現する
        network_dim = np.array([classes * 2, classes * 2])

        # 各ユニットの重みベクトル w_ij を [0,1) の一様乱数で初期化
        self.structure = np.random.random((network_dim[0], network_dim[1], self.m))

        # 初期近傍半径 r_0 はマップの半径程度に設定
        r_init = max(network_dim[0], network_dim[1]) / 2
        # r(t) の減衰定数 τ を決めるためのスケール
        t_cons = self.n / np.log(r_init)

        # ログ取得用のリスト
        list_bmu = []  # 各イテレーションでの BMU の座標
        list_radius = []  # 半径 r(t)
        list_lr = []  # 学習率 η(t)
        list_distance = []  # 入力と BMU の最小距離 ||x - w_bmu||²

        # オンライン学習：各イテレーションで 1 サンプルずつ入力
        for i in range(iterations):
            # 入力ベクトル s ∈ R^m（列ベクトル化）
            s = list_data[i, :].reshape(np.array([self.m, 1]))

            # BMU とその距離を計算
            _, x_bmu, dist = self.best_matching_unit(s)
            list_bmu.append(x_bmu)
            list_distance.append(dist)

            # 時刻 t=i における近傍半径 r(t) と学習率 η(t)
            r = self.change_radius(r_init, i, t_cons)
            l = self.lr_decay(self.learning_rate, i, iterations)
            list_radius.append(r)
            list_lr.append(l)

            # 全ユニット (x, y) について重みベクトルを更新
            for x in range(self.structure.shape[0]):
                for y in range(self.structure.shape[1]):
                    weight = self.structure[x, y, :].reshape(self.m, 1)

                    # BMU との格子距離 d = ||(x,y) - x_bmu||²
                    weight_dist = np.sum((np.array([x, y]) - x_bmu) ** 2)

                    # d <= r² の範囲を「近傍」とみなし更新対象とする
                    if weight_dist <= r**2:
                        # 近傍関数 h(d, r)
                        influence = self.changes(weight_dist, r)

                        # SOM の更新式
                        #   w(t+1) = w(t) + η(t) h(d,r) (s - w(t))
                        # BMU に近いほど h が大きくなり，大きく移動する
                        weight_new = weight + (l * influence * (s - weight))
                        self.structure[x, y, :] = weight_new.reshape(1, self.m)

        list_bmu = np.array(list_bmu)
        return list_bmu, list_radius, list_lr, list_distance

    # 学習後のマップ上での入力の配置を可視化するための関数
    def som_builder(self, target, list_bmu, classes):
        # 「未ソートな入力」を表示するためのランダムな 2D 座標を用意
        data_2_x = np.random.randint(0, 6, self.n)
        data_2_y = np.random.randint(0, 6, self.n)

        # クラスラベルを RGB にエンコード
        target_color = np.zeros((self.n, 3))
        for i, v in enumerate(target):
            if v == 0:
                target_color[i, 0] = 1  # setosa → 赤
            elif v == 1:
                target_color[i, 1] = 1  # versicolor → 緑
            elif v == 2:
                target_color[i, 2] = 1  # virginica → 青

        # ノイズを加えることで、点が完全に重ならないようにする
        x_noise_min = y_noise_min = -0.4
        x_noise_max = y_noise_max = 0.4
        x_noise = (x_noise_max - x_noise_min) * np.random.rand(
            self.n,
        ) + x_noise_min
        y_noise = (y_noise_max - y_noise_min) * np.random.rand(
            self.n,
        ) + y_noise_min

        # SOM 上での BMU 座標にノイズを加えたもの（「ソート済み + ノイズあり」）
        plt_x_noise = list_bmu[:, 0] + x_noise
        plt_y_noise = list_bmu[:, 1] + y_noise

        # ランダム座標に対するノイズ（「未ソート + ノイズあり」）
        x_noise = data_2_x + x_noise
        y_noise = data_2_y + y_noise

        # 凡例用のダミー scatter
        e_legend = [
            plt.scatter(0, 0, c="r", label="setosa"),
            plt.scatter(0, 0, c="g", label="versicolor"),
            plt.scatter(0, 0, c="b", label="virginica"),
        ]

        # 1) 未ソート + ノイズなし：クラス構造なしのランダム配置
        plt.scatter(data_2_x, data_2_y, s=20, c=target_color)
        plt.title(f"{self.n} Inputs unsorted without noise")
        plt.legend(handles=e_legend, loc=1)
        plt.show()

        # 2) 未ソート + ノイズあり
        plt.scatter(x_noise, y_noise, s=20, c=target_color)
        plt.title(f"{self.n} Inputs unsorted with noise")
        plt.legend(handles=e_legend, loc=1)
        plt.show()

        # 3) ソート済み（SOM 上の BMU 座標）：トポロジ保存写像の結果
        plt.scatter(list_bmu[:, 0], list_bmu[:, 1], s=20, c=target_color)
        plt.title(f"{self.n} Inputs sorted without noise")
        plt.legend(handles=e_legend, loc=1)
        plt.show()

        # 4) ソート済み + ノイズあり：クラスタの重なりを視覚的に分かりやすくする
        plt.scatter(plt_x_noise, plt_y_noise, s=20, c=target_color)
        plt.title(f"{self.n} Inputs sorted with noise")
        plt.legend(handles=e_legend, loc=1)
        plt.show()


def main():
    # Iris データセットの読み込み
    data = load_iris()
    inputs = data["data"]  # 4 次元特徴（がく片長・幅，花弁長・幅）
    inputs = (
        inputs / inputs.max()
    )  # 特徴量を [0,1] にスケーリング（SOM 学習を安定させるため）
    target = data["target"]  # クラスラベル（0,1,2）
    classes = 3  # クラス数（マップサイズの設定に利用）
    learning_rate = 0.3  # 初期学習率 η₀

    som = SOM(learning_rate)

    # SOM の学習
    # best_mu: 各イテレーションでの BMU 座標
    # r_data: 近傍半径 r(t) の履歴
    # rate:   学習率 η(t) の履歴
    # squqred_dist: BMU との距離 ||x - w_bmu||² の履歴
    best_mu, r_data, rate, squqred_dist = som.som_training(inputs, classes, 150)

    # 学習後のマップ上での入力配置を可視化
    som.som_builder(target, best_mu, classes)

    # 近傍半径 r(t) の時間変化
    plt.title("radius_evolution")
    plt.xlabel("iterations")
    plt.ylabel("radius_size")
    plt.plot(r_data)
    plt.show()

    # 学習率 η(t) の時間変化
    plt.title("learning_rate_evolution")
    plt.xlabel("iterations")
    plt.ylabel("learning_rate")
    plt.plot(rate)
    plt.show()

    # BMU との距離（量子化誤差）の推移
    #   小さくなっていれば，マップが入力分布に適応していることを意味する
    plt.title("best_matching_unit_3D_distance")
    plt.xlabel("iterations")
    plt.ylabel("smallest_distance_squared")
    plt.plot(squqred_dist)
    plt.show()


if __name__ == "__main__":
    main()
