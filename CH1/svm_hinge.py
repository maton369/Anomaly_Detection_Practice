import numpy as np
import matplotlib.pyplot as plt

# 乱数シードを固定して、毎回同じデータ・学習結果になるようにする
np.random.seed(1)


class SVM:
    """
    線形SVM (Soft-margin SVM) を勾配降下法で素朴に学習するクラス。

    理論的には、以下の最適化問題を解いていることになる：

        最小化：
            1/2 ||w||^2 + λ * Σ_i max(0, 1 - y_i (w·x_i - b))

    ・第1項 1/2 ||w||^2 は L2 正則化項（大きすぎる重みを抑え、マージンを大きく取る）
    ・第2項 max(0, 1 - y_i (w·x_i - b)) はヒンジ損失（マージン違反に対するペナルティ）
    ・λ (lambda_param) は正則化強さのハイパーパラメータ
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        # 学習率（勾配降下法でパラメータをどれだけ更新するか）
        self.lr = learning_rate
        # 正則化パラメータ λ：大きいほど ||w||^2 を小さくしようとする（マージンを広げる方向）
        self.lambda_param = lambda_param
        # 勾配降下の反復回数（エポック数に相当）
        self.n_iters = n_iters
        # 重みベクトル w とバイアス b は後で初期化
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        線形SVMを勾配降下法（厳密にはサブグラディエント降下）で学習する。

        ・X: (n_samples, n_features) の入力ベクトル
        ・y: ラベル（ここでは {1, -1} に変換して使用）

        サンプルごとに以下を行うイメージ：
        - マージン条件 y_i (w·x_i - b) >= 1 が満たされているかどうかを確認
        - 満たされていれば、正則化項のみの勾配で更新
        - 満たされていなければ、正則化項 + ヒンジ損失の勾配で更新
        """
        # サンプル数と特徴量の次元数を取得
        n_samples, n_features = X.shape

        # y を {0, 1} などから {−1, 1} に変換する
        # SVM の理論ではラベルは {−1, 1} を仮定するので、その形式に合わせる
        y_ = np.where(y <= 0, -1, 1)

        # 重みベクトル w を 0 で初期化（原点からのスタート）
        self.w = np.zeros(n_features)
        # バイアス b も 0 からスタート
        self.b = 0

        # n_iters 回だけデータを反復して学習（エポック）
        for _ in range(self.n_iters):
            # 各サンプル x_i について、マージン条件をチェックしながら更新
            for idx, x_i in enumerate(X):
                # マージン条件 y_i (w·x_i - b) >= 1 を計算
                # ここで (w·x_i - b) は決定関数 f(x)
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1

                if condition:
                    # マージン条件を満たしている場合：
                    # → ヒンジ損失は 0 なので、正則化項 1/2 ||w||^2 のみを意識した勾配になる
                    # 勾配は 2 * λ * w （L2正則化の勾配）で、それを引くことで w を小さくする方向へ更新
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # マージン条件を満たしていない場合：
                    # → ヒンジ損失 max(0, 1 - y_i (w·x_i - b)) の勾配が効いてくる
                    #
                    # ヒンジ損失の w に関するサブグラディエントは -y_i x_i
                    # 正則化を含めると、全体の勾配は 2 λ w - y_i x_i
                    # したがって w はその勾配の負方向へ更新する：
                    #   w ← w - lr * (2 λ w - y_i x_i)
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    # バイアス b に関しては、ヒンジ損失のサブグラディエントは y_i
                    # なので b ← b - lr * (- ∂L/∂b) = b - lr * y_i
                    # （符号の取り方は実装に依存するが、ここでは y_i を引く形）
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        """
        学習済みの w, b を用いて、入力 X に対してクラスラベルを予測する。

        線形決定関数：
            f(x) = w·x - b
        を計算し、その符号 sign(f(x)) によって {−1, 1} のクラスを返す。

        決定境界は f(x) = 0 の超平面、つまり w·x - b = 0 で与えられる。
        """
        # 線形の決定関数値 f(x) を計算
        approx = np.dot(X, self.w) - self.b
        # sign によって +1 / -1 を返す
        return np.sign(approx)


# =========================
# データセットの生成部分
# =========================

# データ点の数
n = 40

# X: 2次元特徴量 (n, 2)、y: ラベル (n,)
X, y = np.zeros((n, 2)), np.zeros(n)

# 0〜19番目のサンプルの x 座標（第1成分）を [-15, -14] 付近に配置
X[0:20, 0] = np.random.rand(n // 2) - 15  # 左側のクラス

# 20〜39番目のサンプルの x 座標（第1成分）を [-5, -4] 付近に配置
X[20::, 0] = np.random.rand(n // 2) - 5  # 右側のクラス

# y 座標（第2成分）は [0, 1] の一様乱数
X[:, 1] = np.random.rand(n)

# 最初の2点だけ x 座標を +10 シフト
# → 本来左側のクラスに属するはずの点が、右側のクラス領域に紛れ込む形になる。
#   これは「異常（outlier）」っぽい点として振る舞うので、SVM がどのように
#   マージンを取るかを見る際の良い例になる。
X[0:2, 0] = X[0:2, 0] + 10

# ラベルを設定：
# 0〜19番目のラベルは +1
y[0:20] = np.ones(n // 2)
# 20〜39番目のラベルは -1
y[20::] = -np.ones(n // 2)

# SVM クラスのインスタンスを生成し、データにフィットさせる
clf = SVM()
clf.fit(X, y)

# 学習された重み w とバイアス b を出力
print("w:", clf.w, "b:", clf.b)


def visualize_svm():
    """
    学習済みの SVM による決定境界（超平面）を2次元平面上に可視化する関数。

    ここでは「決定関数 f(x) = w·x - b = 0」を直線として描画している。
    本来のSVM理論では、この直線からの距離（マージン）が最大化されるように
    w, b が学習されている。
    """

    def get_hyperplane_value(x, w, b, offset):
        """
        決定境界（またはマージンの線）の y 座標を計算する補助関数。

        決定境界は 2次元の場合、次の形で表される：

            w0 * x + w1 * y - b + offset = 0

        ここで offset を 0 にすると決定境界そのもの、
        offset を +1, -1 にするとマージン線に相当するように使うことができる。
        今回は offset=0 のみ使用。

        y について解くと：
            y = (-w0 * x + b - offset) / w1
        """
        return (-w[0] * x + b + offset) / w[1]

    # 描画用の Figure / Axes を準備
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # 学習データ点を散布図で表示
    # c=y とすることで、ラベル +1/-1 に応じて色分けされる
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

    # x 軸方向の最小値・最大値（決定境界を描画する範囲）
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    # 決定境界 f(x)=0 に対応する2点の y 座標を求める
    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

    # 決定境界の直線をプロット（赤の破線）
    ax.plot([x0_1, x0_2], [x1_1, x1_2], "r--")

    # y 軸の表示範囲を少し広めに取る
    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()


# 可視化を実行
visualize_svm()
