# -*- coding: utf-8 -*-

import sys

# 自作のグラフ畳み込み層 GraphConv を読み込むためにパスを追加
# Grid 上の特徴量間の関係をグラフとして扱い，GCN で畳み込む構成になっている
sys.path.append("/home/user/Desktop/TKFile/GraphCNN-Origin/code")
from graph_convolution import GraphConv

import math
import time
import numpy as np
import pandas as pd
import statistics

import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping

# ★ ここでは optimizer を文字列 "adam" で指定しているため，個別の Optimizer クラス import は不要
# from keras.optimizers import adam, RMSprop, SGD, Adagrad, Adadelta
from keras.regularizers import l2, l1

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

###### Setting ####################################################################################################
# 乱数シードを固定して，再現性を確保
seed_val = 1984
np.random.seed(seed_val)

point = []
start = time.time()
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

###### Data Transform #############################################################################################
# 学習用・評価用データの読み込み
# ここでは 0 列目が目的変数，1 列目以降が説明変数と仮定している
am_loc1 = "/home/user/Desktop/GridWorks/Dataset/train_data_1_2.csv"
am_loc2 = "/home/user/Desktop/GridWorks/Dataset/test_data_1_2.csv"
data_x = np.array(pd.read_csv(open(am_loc1, "r")))
data_y = np.array(pd.read_csv(open(am_loc2, "r")))

# もし特徴量のスケール差が大きい場合は，下記のように正規化することも想定されている
# data_x_max = data_x.max(0),
# data_y_max = data_y.max(0)
# data_x = data_x / data_x_max
# data_y = data_y / data_y_max

# 説明変数 (X) と目的変数 (y) に分割
X_train, y_train = data_x[:, 1:], data_x[:, 0]
X_test, y_test = data_y[:, 1:], data_y[:, 0]

###### Correlation ################################################################################################
# 特徴量間の相関に基づいてグラフ構造を構成するパート
# 各特徴量をノードとみなし，相関が高い上位 num_neighbors 個を隣接ノードとして扱う
num_neighbors = 2

# 相関係数行列を計算し，絶対値を取ることで「線形関係の強さ」のみを見ている
corr_mat = np.array(
    normalize(
        np.abs(np.corrcoef(X_train.transpose())),  # 相関係数行列 (特徴量 x 特徴量)
        norm="l1",
        axis=1,  # 各ノードの隣接確率が 1 になるよう L1 正規化
    ),
    dtype="float64",
)

# 各行ごとに相関の高いインデックス上位 num_neighbors 個を取り出し，
# それを GraphConv に渡す neighbors_ix_mat として利用する
graph_mat = np.argsort(corr_mat, 1)[:, -num_neighbors:]

###### GaussianKernel #############################################################################################
# 上の相関ベースのグラフ構成の代わりに，ガウスカーネルに基づく類似度から
# 近傍ノードを決めるオプション（コメントアウト中）
# 理論的には「特徴量空間上のユークリッド距離が近いノード同士を結ぶグラフ」を作ることに相当

# X_trainT = X_train.T
# row  = X_trainT.shape[0]
# kernel_mat = np.zeros(row * row).reshape(row, row)
#
# sigma = 1
# num_neighbors = 6
# for i in range(row):
#     for j in range(row):
#         # ガウスカーネル: exp(-||xi - xj||^2 / (2 sigma^2))
#         kernel_mat[i, j] = math.exp( - (np.linalg.norm(X_trainT[i] - X_trainT[j]) ** 2) / (2 * sigma ** 2))
# # 類似度が高い上位 num_neighbors ノードを隣接ノードとして採用
# graph_mat  = np.argsort(kernel_mat, 1)[:,-num_neighbors:]

###### Learning ###################################################################################################
# 学習のエポック数
epoch = 800
epochs = np.arange(epoch)

# 各エポックでの Train/Test RMSE を格納する配列
# 初期値として大きめの値 (20) を入れておく
results_train = np.ones(epoch) * 20
results_test = np.ones(epoch) * 20

# ミニバッチサイズ
batch_size = 50
# 全結合層のユニット数（GCN 出力を集約した後の隠れ層）
num_hidden = 75
# 各 GraphConv 層のフィルタ数（= 出力チャネル数）
filters_1 = 24
filters_2 = 24
filters_3 = 21
filters_4 = 22
filters_5 = 20

# モデル定義
# ここでは
# GraphConv × 5 → Flatten → Dense → Dense(1)
# という 1 次元グラフ畳み込みネットワーク + MLP による回帰モデルとなっている
model = Sequential()

# 1 層目のグラフ畳み込み
# input_shape = (特徴量数, 1) とし，各特徴量をノードとして扱う 1 次元グラフ
# neighbors_ix_mat と num_neighbors により，
# 各ノードがどのノードとメッセージパッシングするかが決まる
model.add(
    GraphConv(
        filters=filters_1,
        neighbors_ix_mat=graph_mat,
        num_neighbors=num_neighbors,
        activation="relu",
        input_shape=(X_train.shape[1], 1),
    )
)
model.add(BatchNormalization())  # 各チャネルを正規化し，学習を安定させる

# model.add(Dropout(0.25))  # 過学習抑制用の Dropout（現在は無効）

# 2 層目のグラフ畳み込み
model.add(
    GraphConv(
        filters=filters_2,
        neighbors_ix_mat=graph_mat,
        num_neighbors=num_neighbors,
        activation="relu",
    )
)
model.add(BatchNormalization())
# model.add(Dropout(0.25))

# 3 層目のグラフ畳み込み
model.add(
    GraphConv(
        filters=filters_3,
        neighbors_ix_mat=graph_mat,
        num_neighbors=num_neighbors,
        activation="relu",
    )
)
# model.add(BatchNormalization())
# model.add(Dropout(0.25))

# 4 層目のグラフ畳み込み
model.add(
    GraphConv(
        filters=filters_4,
        neighbors_ix_mat=graph_mat,
        num_neighbors=num_neighbors,
        activation="relu",
    )
)
# model.add(BatchNormalization())
# model.add(Dropout(0.25))

# 5 層目のグラフ畳み込み
model.add(
    GraphConv(
        filters=filters_5,
        neighbors_ix_mat=graph_mat,
        num_neighbors=num_neighbors,
        activation="relu",
    )
)
model.add(BatchNormalization())
# model.add(Dropout(0.25))

# グラフ畳み込み層の出力は (ノード数, チャネル数) なので，
# Flatten で 1 次元ベクトルに変換して全結合層へ渡す
model.add(Flatten())

# L2 正則化付きの全結合層
# ここで特徴量の高次な組み合わせを学習し，最後にスカラー回帰値へマッピングする
model.add(
    Dense(
        num_hidden,
        kernel_regularizer=l2(0.01),
    )
)
model.add(BatchNormalization())
model.add(Activation("relu"))
# model.add(Dropout(0.1))

# 出力層：ユニット数 1 の線形回帰
model.add(Dense(1, kernel_regularizer=l2(0.01)))
# model.add(Dropout(0.1))

model.summary()

# 損失関数として平均二乗誤差 (MSE) を使用し，最適化は Adam
# 理論的には，「RMSE を最小化する回帰問題」を MSE で近似的に解いていることになる
model.compile(loss="mean_squared_error", optimizer="adam")

# エポックごとの学習ループ
for i in epochs:
    # 学習データとテストデータから，エポックごとに半分ずつサンプリングして
    # 「動的な train/test 分割」を行っている
    # → データが少ない場合に，より多くの組み合わせで性能を評価したい意図と思われる
    ramdom1 = np.random.choice(len(data_x), len(data_x) // 2, replace=False)
    ramdom2 = np.random.choice(len(data_y), len(data_y) // 2, replace=False)
    data_xb = data_x[ramdom1]
    data_yb = data_y[ramdom2]
    X_train, y_train = data_xb[:, 1:], data_xb[:, 0]
    X_test, y_test = data_yb[:, 1:], data_yb[:, 0]

    # GraphConv が (サンプル数, ノード数, 1) 形式を想定しているため，
    # reshape で特徴量軸を「ノード」として明示する
    model.fit(
        X_train.reshape(X_train.shape[0], X_train.shape[1], 1),
        y_train,
        epochs=1,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.25,  # 学習データ内でさらに 25% を検証用に分割
        verbose=0,
    )

    # 学習データ・検証データに対する予測を計算
    pred_train = np.array(
        model.predict(
            X_train.reshape(X_train.shape[0], X_train.shape[1], 1), batch_size=10
        )
    ).flatten()
    pred_test = np.array(
        model.predict(
            X_test.reshape(X_test.shape[0], X_test.shape[1], 1), batch_size=10
        )
    ).flatten()

    # RMSE (二乗誤差の平方根) を評価指標として利用
    RMSE_train = np.sqrt(mean_squared_error(y_train, pred_train))
    RMSE_test = np.sqrt(mean_squared_error(y_test, pred_test))
    # もし y を別スケールに戻して評価したい場合は，下記のようにスケーリングする想定
    # RMSE_train = np.sqrt(mean_squared_error(y_train * 80, pred_train * 80))
    # RMSE_test  = np.sqrt(mean_squared_error(y_test  * 80, pred_test  * 80))

    # エポックごとの RMSE を記録
    results_train[i] = RMSE_train
    results_test[i] = RMSE_test
    Min_RMSE_test = results_test.min()

    # 現時点でのテスト RMSE が最小値を更新したかどうかでログ出力を切り替え
    if RMSE_test == Min_RMSE_test:
        print(
            "Epoch: %d, Train_RMSE: %.4f, Min_RMSE_test: %4f"
            % (i, RMSE_train, RMSE_test)
        )
    else:
        print(
            "Epoch: %d, Train_RMSE: %.4f, RMSE_test: %4f" % (i, RMSE_train, RMSE_test)
        )

    ###### RealTime RMSE Plot ######
    # 以下は学習中に RMSE の推移をリアルタイムでプロットするためのコード（現在は無効）
    # plt.plot(epochs, results_train, color='blue',  linestyle='--', )
    # plt.plot(epochs, results_test,  color='green', linestyle='--', )
    # plt.ylim(4, 12)
    # plt.pause(0.005)
    # plt.cla()

###### Results #########################################################################################################
# 最終エポック後の予測と RMSE を再計算
pred_train = np.array(
    model.predict(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), batch_size=5)
).flatten()
pred_test = np.array(
    model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), batch_size=5)
).flatten()
RMSE_train = np.sqrt(mean_squared_error(y_train, pred_train))
RMSE_test = np.sqrt(mean_squared_error(y_test, pred_test))
process_time = time.time() - start

# テストデータの最後の 10 サンプルについて，
# 真値と予測値を表示して，モデルの挙動をざっくり確認
print(np.round(y_test[-10:,]))
print(np.round(pred_test[-10:,]))

print("Min_Train-RMSE: ", results_train.min())
print("Min_Test-RMSE:  ", results_test.min())
print("process_time:   ", process_time)

###### RMSE Comparing Figure ###########################################################################################
# RMSE の推移を平滑化して可視化する
num = 20
smooth = np.ones(num) / num
smooth1 = np.convolve(results_train, smooth, mode="same")
smooth2 = np.convolve(results_test, smooth, mode="same")

plt.close()
plt.figure(figsize=(12, 9))
# 元の散布（各エポックの RMSE）
plt.plot(
    epochs,
    results_train,
    color="blue",
    linestyle="None",
    marker=".",
    markersize=1,
    label="Train",
)
plt.plot(
    epochs,
    results_test,
    color="green",
    linestyle="None",
    marker=".",
    markersize=1,
    label="Test",
)

# 移動平均による平滑化曲線（局所的なトレンドの把握用）
plt.plot(
    epochs[11 : epoch - 11,],
    smooth1[11 : epoch - 11,],
    color="blue",
    linewidth=3,
    label="Train smooth",
)
plt.plot(
    epochs[11 : epoch - 11,],
    smooth2[11 : epoch - 11,],
    color="green",
    linewidth=3,
    label="Test smooth",
)

plt.xlabel("Epoch", fontsize=16)
plt.ylabel("RMSE", fontsize=16)
plt.ylim(0, 15)
plt.legend(fontsize=15)
plt.show()
