import tensorflow as tf
from keras import backend as K
from keras import activations, regularizers, constraints, initializers

# ★ Keras 3 / tf.keras では InputSpec は keras.engine ではなく keras.layers から import する
from keras.layers import Layer, InputSpec


class GraphConv(Layer):
    """Convolution operator for graphs.

    元の実装は Theano バックエンド前提だったが，この版では TensorFlow / tf.keras でも
    動作するように書き換えている（tensordot や gather を TensorFlow 実装に変更）。

    1 次元畳み込み conv1d をグラフに一般化したものになっており，
    「各ノードの近傍ノード群」から特徴量を集約して出力チャネルを生成する。

    # 入出力の形状
        入力:  (batch_size, features, input_dim)
        出力:  (batch_size, features, filters)

        ・features: グラフのノード数（もともとの特徴量次元）
        ・input_dim: 各ノードが持つチャネル数
        ・filters: 畳み込み後のチャネル数
    """

    def __init__(
        self,
        filters,
        num_neighbors,
        neighbors_ix_mat,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        # ここでは Theano チェックを削除している
        # （Keras 3 / tf.keras では TensorFlow バックエンドが前提のため）
        #
        # if K.backend() != "theano":
        #     raise Exception("GraphConv Requires Theano Backend.")

        # 古い Keras では input_dim だけ指定してくる場合があるので，
        # そのときは input_shape に変換して親クラスに渡す
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)

        # 親クラス Layer の初期化
        super(GraphConv, self).__init__(**kwargs)

        # 出力チャネル数（グラフ畳み込みのフィルタ数）
        self.filters = filters
        # 各ノードが参照する近傍ノードの数（1 次近傍の数）
        self.num_neighbors = num_neighbors
        # 形状 (num_features, num_neighbors) の整数行列
        # neighbors_ix_mat[i, j] = 「特徴 i が j 番目の近傍として参照するノードのインデックス」
        self.neighbors_ix_mat = tf.convert_to_tensor(neighbors_ix_mat, dtype=tf.int32)

        # 活性化関数や正則化，制約などを Keras のユーティリティから取得
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = (
            activity_regularizer  # Layer 側で処理されるのでそのまま保持
        )
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        # 入力テンソルは 3 次元 (batch, features, input_dim) を仮定
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        # 入力の最後の次元が「各ノードの特徴量次元数」
        input_dim = int(input_shape[2])

        # カーネルの形状:
        # (近傍の数, 入力チャネル数, 出力チャネル数)
        # すなわち，「各近傍ノードから入力チャネル → 出力チャネルへの写像」をまとめた 3 階テンソル
        kernel_shape = (self.num_neighbors, input_dim, self.filters)

        # 学習パラメータ W（グラフ畳み込みカーネル）の登録
        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            # 出力チャネルごとのバイアス項
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        self.built = True

    def call(self, x):
        # x の形状は (batch, features, input_dim)
        #
        # neighbors_ix_mat（形状: (features, num_neighbors)）を用いて
        # 各ノードの近傍ノードをまとめて取得する。
        # TensorFlow では Theano の x[:, idx, :] 風の高度なインデックス指定ができないため，
        # tf.gather を用いて axis=1（features 軸）に沿って取り出す。
        #
        # 結果の形状:
        #   x_expanded: (batch, features, num_neighbors, input_dim)
        x_expanded = tf.gather(x, self.neighbors_ix_mat, axis=1)

        # tensordot により，x_expanded と kernel の積を一気に計算する
        #
        # x_expanded: (batch, features, num_neighbors, input_dim)
        # kernel    : (num_neighbors, input_dim, filters)
        #
        # [[2,3],[0,1]] という指定は
        #   x_expanded の軸 (num_neighbors, input_dim)
        #   kernel     の軸 (num_neighbors, input_dim)
        # を縮約（総和）することを意味する。
        #
        # 各ノードごとに「近傍 num_neighbors 個 × 入力チャネル input_dim」の線形結合を取り，
        # 出力チャネル filters を得る，というグラフ畳み込みの式
        #   h_i^{(out)} = Σ_{j∈N(i)} Σ_c x_j^{(c)} * W_{(i,j,c,:)}
        # をテンソル積で実装しているイメージになる。
        output = tf.tensordot(x_expanded, self.kernel, axes=[[2, 3], [0, 1]])

        # バイアス項を付加（チャネル方向にだけ足すため，(1,1,filters) に reshape してブロードキャスト）
        if self.use_bias:
            output += tf.reshape(self.bias, (1, 1, self.filters))

        # 非線形活性化関数を適用
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        # features の数は入力と同じで，チャネル数だけ filters に変化する
        return (input_shape[0], input_shape[1], self.filters)
