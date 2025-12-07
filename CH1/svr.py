import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf


class SVR(object):
    """
    ε-不感誤差（epsilon-insensitive loss）を使った線形 SVR (Support Vector Regression) の簡易実装クラス。
    TensorFlow 2 系で動作するように Session / placeholder は使わず、eager execution + GradientTape で最適化する。

    理論的には次の最適化問題を解いているイメージになる：
    $$L(w,b)=\frac12\lVert w\rVert^2 + \frac1n\sum_{i=1}^n\max\bigl(0,\lvert w^\top x_i + b - y_i\rvert - \epsilon\bigr)$$

    ・第1項 $$\frac12\lVert w\rVert^2$$ は重みベクトルの L2 正則化（モデルの複雑さペナルティ）
    ・第2項は ε-不感損失で、「誤差が ε を超えた部分だけ」を線形ペナルティとして加算する。
      → 誤差が ±ε のチューブ内に収まっている点は損失 0 となり、サポートベクトル以外は目的関数に寄与しない。
    """

    def __init__(self, epsilon=0.5):
        # ε-不感帯の幅を設定（この幅以内の誤差は損失にカウントされない）
        self.epsilon = float(epsilon)
        # モデルパラメータ（学習後にセットされる）
        self.W = None
        self.b = None

    def fit(self, X, y, epochs=100, learning_rate=0.1):
        """
        勾配降下法（TensorFlow 2 の GradientTape による自動微分）を使って
        SVR のパラメータ (W, b) を学習する。
        """
        # NumPy 配列に変換しておく
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # 入力が 1 次元なら (n_samples, 1) に reshape する
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, feature_len = X.shape

        # Tensor に変換（学習中はずっと固定なので constant で良い）
        X_tf = tf.constant(X, dtype=tf.float32)
        y_tf = tf.constant(y, dtype=tf.float32)

        # モデルパラメータ W（重み）と b（バイアス）をランダム初期化
        # W: (feature_len, 1), b: (1,)
        self.W = tf.Variable(tf.random.normal(shape=(feature_len, 1), dtype=tf.float32))
        self.b = tf.Variable(tf.random.normal(shape=(1,), dtype=tf.float32))

        # 最適化アルゴリズム（確率的勾配降下）
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        # 学習ループ
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                # 予測値 y_pred = XW + b
                y_pred = tf.matmul(X_tf, self.W) + self.b  # (n_samples, 1)

                # L2 正則化項 1/2||W||^2 （tf.norm は L2 ノルム）
                reg_term = 0.5 * tf.norm(self.W) ** 2

                # ε-不感損失項: max(0, |y_pred - y| - ε) の平均
                abs_err = tf.abs(y_pred - y_tf)
                eps_loss = tf.maximum(0.0, abs_err - self.epsilon)
                data_term = tf.reduce_mean(eps_loss)

                # 全体の損失
                loss = reg_term + data_term

            # W, b に関する勾配を計算
            grads = tape.gradient(loss, [self.W, self.b])

            # 勾配降下でパラメータを更新
            optimizer.apply_gradients(zip(grads, [self.W, self.b]))

            # 学習の進み具合を表示
            print("{}/{}: loss: {:.6f}".format(epoch + 1, epochs, float(loss.numpy())))

        return self

    def predict(self, X, y=None):
        """
        学習済みモデルを使って入力 X に対する予測値を計算する。
        """
        if self.W is None or self.b is None:
            raise RuntimeError(
                "モデルがまだ学習されていません。先に fit() を呼び出してください。"
            )

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_tf = tf.constant(X, dtype=tf.float32)
        y_pred = tf.matmul(X_tf, self.W) + self.b

        # NumPy 配列で返す
        return y_pred.numpy().reshape(-1)


# =========================
# データ生成と学習のデモ
# =========================

# 入力 x を [0, 5] の範囲で 20 点に等間隔に生成
x = np.linspace(start=0, stop=5, num=20)

# 真の一次関数の傾きと切片
m = 2.0
c = 1.0

# 真の回帰直線 y = m x + c
y = m * x + c

# ガウスノイズを加えて「観測値らしい」ばらつきを持たせる
y += np.random.normal(size=(len(y),))

# 観測データを散布図としてプロット
plt.figure()
plt.plot(x, y, "x")

# ε-不感帯の幅 ε = 0.2 の SVR モデルを生成
# → 誤差が ±0.2 の範囲内であれば損失 0 とみなす
model = SVR(epsilon=0.2)

# SVR モデルを学習
model.fit(x, y, epochs=100, learning_rate=0.05)

# 学習データと SVR による予測直線を重ねて描画
y_pred = model.predict(x)
plt.plot(x, y, "x", x, y_pred, "-")
plt.legend(["actual", "prediction"])
plt.xlabel("x")
plt.ylabel("y")
plt.title("SVR with epsilon-insensitive loss")
plt.show()
