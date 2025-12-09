# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
from numpy.random import rand, multivariate_normal
from sklearn import datasets

# Iris データセットの読み込み
iris = datasets.load_iris()
# 花弁の長さ (petal length) と幅 (petal width) を取り出す
nagasa = iris.data[:, 2]
haba = iris.data[:, 3]

# 今回は「花弁の長さ・幅」の 2 次元データ X を対象に PCA を行う
X = iris.data[:, 2:4]
N = 150  # サンプル数


def centering(X, N):
    """
    データ行列 X を「平均 0」に中心化する関数。

    理論的には，中心化は
        X_c = H X
    という形で行列 H を左から掛ける操作になり，
        H = I - (1/N) * 11^T
    を「中心化行列 (centering matrix)」と呼ぶ。

    - I : N×N の単位行列
    - 1 : N 次元の全ての成分が 1 のベクトル
    - 11^T : 全成分が 1 の N×N 行列

    この H は
        H 1 = 0
    を満たし，任意のデータ行列 X から「各次元の平均成分」を引き去る役割を持つ。
    PCA の標準的な前処理として必須のステップ。
    """
    # 中心化行列 H を構成
    H = np.eye(N) - 1.0 / N * np.ones([N, N])
    # H X を計算して中心化し，2 次元の DataFrame として返す
    return DataFrame(np.dot(H, X), columns=["x", "y"])


# 中心化を実行
df_center = centering(X, N)

# 各クラス（3 種類のアヤメ）ごとに色分けして散布図を描画
plt.scatter(df_center[0:50]["x"], df_center[0:50]["y"], c="blue", marker="o")
plt.scatter(df_center[50:100]["x"], df_center[50:100]["y"], c="red", marker="o")
plt.scatter(df_center[100:150]["x"], df_center[100:150]["y"], c="green", marker="o")
plt.show()


def PCA(X):
    """
    2 次元データに対して主成分分析 (PCA) を行い，
    固有ベクトル（主成分方向）を返す関数。

    PCA の理論：
    - 中心化済みデータ X (N×d) を用意する
    - 散布行列（共分散行列から定数係数を抜いたもの）
        S = X^T X
      を計算する
    - S は対称行列かつ半正定値なので，固有値分解が可能
        S v_k = λ_k v_k
    - 固有値 λ_k が大きいほど，その固有ベクトル v_k の向きに
      データの分散が大きいことを意味する
    - よって，固有値の大きい順に固有ベクトルを並べると，
      それが「第 1 主成分・第 2 主成分 …」の方向になる
    """
    # 標本の散布行列 C = X^T X を計算
    # （共分散行列は 1/N * X^T X だが，1/N は固有ベクトルの向きには影響しない）
    C = np.dot(X.T, X)

    # 固有値 w と正規化された固有ベクトル v を計算
    # 対称行列用の eigh を使うことで数値的に安定した分解を行う
    w, v = np.linalg.eigh(C)

    # 固有値を降順に並び替える（eigh は昇順で返すため）
    index = np.argsort(w)[::-1]

    # 固有ベクトルを固有値の大きい順に並べ替え
    # T_pca[0,:] が第 1 主成分の方向ベクトル，
    # T_pca[1,:] が第 2 主成分の方向ベクトルになる
    T_pca = v[index]
    return T_pca


# PCA を実行し，第 1・第 2 主成分方向を取得
T_pca = PCA(df_center)


def f_pca(x):
    """
    主成分方向に対応する「直線の方程式」を返す関数。

    各主成分の固有ベクトルを
        v = (v_x, v_y)
    とすると，原点を通るその方向の直線は
        y = (v_y / v_x) * x
    と表せる（v_x ≠ 0 と仮定）。

    ここでは
    - y1: 第 1 主成分方向に沿った直線
    - y2: 第 2 主成分方向に沿った直線
    を返している。
    """
    # 第 1 主成分：T_pca[0] = (v1_x, v1_y)
    y1 = T_pca[0][1] / T_pca[0][0] * x
    # 第 2 主成分：T_pca[1] = (v2_x, v2_y)
    y2 = T_pca[1][1] / T_pca[1][0] * x
    return y1, y2


# 主成分軸を描画するための x 座標（中心化済みデータの範囲を少しスケール）
linex = np.arange(df_center["x"].min() * 0.1, df_center["x"].max() * 0.1, 0.01)

# 第 1・第 2 主成分に対応する直線の y 座標を計算
liney1, liney2 = f_pca(linex)

# 主成分軸を描画
plt.plot(linex, liney1, color="red", label="1st principal axis")
plt.plot(linex, liney2, color="blue", label="2nd principal axis")

# データ点も再度プロットして，主成分軸との関係を可視化
plt.scatter(df_center[0:50]["x"], df_center[0:50]["y"], c="blue", marker="o")
plt.scatter(df_center[50:100]["x"], df_center[50:100]["y"], c="red", marker="o")
plt.scatter(df_center[100:150]["x"], df_center[100:150]["y"], c="green", marker="o")
plt.legend()
plt.show()
