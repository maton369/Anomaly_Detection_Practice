import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import copy
from sklearn.datasets import load_iris


class Kmeans:
    """
    K-means クラスタリングを素朴に実装したクラス。

    - 入力ベクトル集合 {x_i}_{i=1}^n を K 個のクラスタに分割し，
      各クラスタの「重心」 μ_k を求めることが目的。
    - 目的関数は「クラスタ内平方和（SSE）」を最小化することに対応する。

        SSE = Σ_k Σ_{x_i ∈ C_k} || x_i - μ_k ||^2

    - アルゴリズムは，以下の 2 ステップを繰り返すことで実現される：

        1. 割り当てステップ（E-step）:
           各点 x_i を，最も近い重心 μ_k に割り当てる。
        2. 重心更新ステップ（M-step）:
           各クラスタ C_k に属する点の平均で新しい μ_k を計算する。
    """

    def __init__(self, inputs, K=3):
        # 入力データ行列 X（形状: (サンプル数 n, 特徴次元 d)）
        self.inputs = inputs
        # サンプル数 n
        self.n = inputs.shape[0]
        # クラスタ数 K
        self.K = K
        # 各サンプルが属するクラスタ番号（0 〜 K-1）を格納する配列
        self.clusters = np.zeros(self.n)
        # 各サンプルと各クラスタ中心とのユークリッド距離を格納する行列
        # distances[i, k] = サンプル i と重心 k の距離
        self.distances = np.zeros((self.n, self.K))
        # 初期クラスタ中心をランダムに生成
        self.centers = self.cal_centers()
        # 1 つ前の反復でのクラスタ中心（収束判定用）
        self.centers_old = np.zeros(self.centers.shape)
        # 現在のクラスタ中心
        self.centers_new = self.centers.copy()

    def cal_centers(self):
        """
        クラスタ中心の初期値を計算する関数。

        - 各次元ごとに，入力データの平均・標準偏差を用いて
          ガウス分布 N(mean, std^2) から K 個サンプルするイメージ。
        - こうすることで，「入力データと同程度のスケール」に
          ばらついた初期重心を得ることができる。
        """
        return np.random.randn(self.K, self.inputs.shape[1]) * self.inputs.std(
            axis=0
        ) + self.inputs.mean(axis=0)

    def update_centers(self):
        """
        K-means のメインループを回してクラスタ中心を更新する。

        - while ループの中で
            1) 各点と各重心の距離を計算
            2) 最も近い重心へ割り当て（式(83)）
            3) 各クラスタの平均をとって重心を更新（式(84)）
          を繰り返し，重心が変化しなくなったところで収束とみなす。
        """
        # centers_new と centers_old の差が 0（= 変化なし）になるまで繰り返す
        while self.error() != 0:
            # --- 割り当てステップ（E-step） ---
            for i in range(self.K):
                # self.inputs: (n, d), self.centers[i]: (d,)
                # 各サンプルと i 番目の重心とのユークリッド距離 ||x_i - μ_i||
                self.distances[:, i] = np.linalg.norm(
                    self.inputs - self.centers[i], axis=1
                )

            # 式(83)に関する計算
            # 各サンプル i について，もっとも距離が小さい重心 k を選び，
            # そのインデックスをクラスタラベルとして格納する。
            #
            #   c_i = argmin_k ||x_i - μ_k||^2
            #
            self.clusters = np.argmin(self.distances, axis=1)

            # 今回の反復に入る前の中心を保存しておき，収束判定に使う
            self.centers_old = self.centers_new.copy()

            # --- 重心更新ステップ（M-step） ---
            for i in range(self.K):
                # 式(84)に関する計算
                # クラスタ i に属する点の集合 C_i に対し，
                # その単純平均を取ることで新しい重心 μ_i を計算する。
                #
                #   μ_i = (1 / |C_i|) Σ_{x_j ∈ C_i} x_j
                #
                # ※ self.inputs[self.clusters == i] で C_i を取り出している。
                self.centers_new[i] = np.mean(self.inputs[self.clusters == i], axis=0)

            # NOTE: 現在のコードでは self.centers を更新していないため，
            #       次の反復で距離計算に古い中心 self.centers が使われてしまう。
            #       本来は self.centers = self.centers_new.copy() などで
            #       上書きする必要がある点に注意。
            self.centers = self.centers_new.copy()

        # 収束したクラスタ中心を返す
        return self.centers_new

    def error(self):
        """
        収束判定に使う「中心の変化量」を計算する関数。

        - 新旧の重心行列の差の Frobenius ノルム
              ||M_new - M_old||_F
          を返す。
        - これが 0 になれば「重心がこれ以上動かない」とみなせるので，
          アルゴリズムは収束したと判断できる。
        """
        return np.linalg.norm(self.centers_new - self.centers_old)


def main():
    # Iris データセットの読み込み
    data = load_iris()
    # 4 次元の実数値特徴量 (n, 4)
    inputs = data["data"]
    # 真のクラスラベル（0: setosa, 1: versicolor, 2: virginica）
    target = data["target"]

    # K=3 で K-means を実行（Iris のクラス数に合わせている）
    kmeans = Kmeans(inputs)
    centers = kmeans.update_centers()

    # 結果の可視化：ここでは 1 次元目と 2 次元目の平面に投影して表示
    plt.figure(figsize=(15, 7))
    colors = ["orange", "blue", "green"]

    # 各サンプルを真のクラスラベルに応じて色分けしてプロット
    for i in range(inputs.shape[0]):
        plt.scatter(
            inputs[i, 0],
            inputs[i, 1],
            s=50,
            color=colors[int(target[i])],
        )

    # 学習されたクラスタ中心を赤い星マーカーで重ねて表示
    plt.scatter(centers[:, 0], centers[:, 1], marker="*", c="r", s=300)
    plt.title("K-means clustering on Iris (1st vs 2nd feature)")
    plt.xlabel("sepal length")
    plt.ylabel("sepal width")
    plt.show()


if __name__ == "__main__":
    main()
