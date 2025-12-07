import numpy as np
import matplotlib.pyplot as plt


class DecisionTree:
    def __init__(self, split_minimum=2, depth_maximum=100, _lambda=0.1, gamma=0.1):
        # 1 ノードを分割するために必要な最小サンプル数
        self.split_minimum = split_minimum
        # 木の最大深さ（大きいほど表現力は高くなるが過学習しやすい）
        self.depth_maximum = depth_maximum
        # L2 正則化項の重み λ（XGBoost の leaf weight 正則化に対応させたパラメータ）
        self._lambda = _lambda
        # 葉ノード数に対するペナルティ γ（葉を増やしすぎることへのペナルティ）
        self.gamma = gamma

    def tree_arrangement(self, inputs, node):
        """
        1 サンプルの入力値に対して，学習済み決定木から予測値を取り出す再帰関数。

        ・葉ノード（val が None でない）ならその値を返す。
        ・内部ノードなら「inputs <= しきい値 thrs」かどうかで左右の子ノードへ降りる。
        """
        if node.val is not None:
            # 葉ノード：格納された値（回帰値）を返す
            return node.val
        # しきい値以下なら左部分木へ
        if inputs <= node.thrs:
            return self.tree_arrangement(inputs, node.left)
        # しきい値より大きければ右部分木へ
        return self.tree_arrangement(inputs, node.right)

    def tree_growth(self, inputs, target, depth=0):
        """
        決定木を再帰的に成長させる。

        本来の XGBoost では「勾配 g, ヘッセ行列 h の和」を使った分割評価
        $$\text{Gain}=\frac{1}{2}\left(\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{G^2}{H+\lambda}\right)-\gamma$$
        を最大化するが，
        この実装ではその代わりに「正則化付きの分散（っぽい量）」を最小化することで
        分割の良さを評価している簡略版になっている。
        """
        # このノードに含まれるサンプル数
        samples = inputs.shape[0]

        # 終端条件：最大深さ or サンプル不足のときは葉ノードを作る
        if depth >= self.depth_maximum or samples < self.split_minimum:
            # 回帰問題なので，このノードでは target の平均値を予測値として持つ
            return Tree_Node(val=np.mean(target))

        # 1 次元入力を前提に，取りうるユニークな値をすべて分割候補しきい値とする
        thresholds = np.unique(inputs)
        # 「最良」の分割（最小の variance 指標）を探すために，大きな初期値を置いておく
        best_variance = 1000

        for th in thresholds:
            # 左：x <= th, 右：x > th となるインデックス集合
            idx_left = np.where(inputs <= th)
            idx_right = np.where(inputs > th)

            # どちらか一方が空なら分割として無効なので，大きなコストを与える
            if len(idx_left[0]) == 0 or len(idx_right[0]) == 0:
                variance = 999
            else:
                # ---- 左右ノードの「正則化付き分散」に対応する量を計算 ----
                # 残差平方和 / (サンプル数 + λ) の平方根としていて，
                # XGBoost の leaf weight 解析解に出てくる denominator (n + λ) を意識した形になっている。
                var_left = np.sqrt(
                    np.sum((target[idx_left] - target[idx_left].mean()) ** 2)
                    / (target[idx_left].shape[0] + self._lambda)
                )
                var_right = np.sqrt(
                    np.sum((target[idx_right] - target[idx_right].mean()) ** 2)
                    / (target[idx_right].shape[0] + self._lambda)
                )

                # さらに葉ノード数に比例したペナルティ γ·|R_j| を足すことで
                # XGBoost の「葉を増やすほどコストが増える」正則化を模倣している
                # （ここでは左右ノードのサンプル数に γ を掛けて加算し，平均を取っている）
                variance = (
                    var_left
                    + target[idx_left].shape[0] * self.gamma
                    + var_right
                    + target[idx_right].shape[0] * self.gamma
                ) / 2

            # 今のしきい値の方が「コストが小さい」（＝分割の質が高い）なら更新
            if variance < best_variance:
                index_left = idx_left
                index_right = idx_right
                best_variance = variance
                threshhold_best = th

        # best_variance が 999 のままということは，有効な分割が見つからなかったということ
        if best_variance == 999:
            # その場合は葉ノードにして，平均値を返す
            return Tree_Node(val=np.mean(target))

        # 最良の分割に基づいて左右の部分集合に対して再帰的に木を成長させる
        left_node = self.tree_growth(inputs[index_left], target[index_left], depth + 1)
        right_node = self.tree_growth(
            inputs[index_right], target[index_right], depth + 1
        )
        # 現在のノードは内部ノードとして，しきい値と左右の子ノードを持つ
        return Tree_Node(threshhold_best, left_node, right_node)

    def fit(self, inputs, target):
        """
        決定木の学習インターフェース。
        root_node を構築することで 1 本の回帰木を得る。
        """
        self.root_node = self.tree_growth(inputs, target)

    def predict(self, inputs):
        """
        学習済み決定木を用いて予測値を計算する。
        各入力値ごとに root から葉まで降りて val を返す。
        """
        return np.array(
            [self.tree_arrangement(input_, self.root_node) for input_ in inputs]
        )


class Tree_Node:
    """
    決定木の 1 つのノードを表すクラス。

    ・内部ノード: thrs に分割しきい値，left/right に子ノードへの参照，val は None
    ・葉ノード: val に予測値（回帰値）を格納し，thrs/left/right は None
    """

    def __init__(self, thrs=None, left=None, right=None, *, val=None):
        self.thrs = thrs
        self.left = left
        self.right = right
        self.val = val


class XGBoost:
    """
    XGBoost 風の勾配ブースティング回帰モデルの簡易実装。

    本家 XGBoost では二階微分（ヘッセ行列）まで使った二次近似に基づき，
    各ノードの分割ゲインや葉の重みを解析的に計算するが，
    この実装では
      ・leaf 重みの L2 正則化（λ）
      ・葉の数に対するペナルティ（γ）
    を DecisionTree 側の分割コストに組み込んだうえで，
    勾配ブースティング的に「残差を近似する木を足していく」
    という構造になっている。
    """

    def __init__(
        self,
        t_numbers=5,
        depth_maximum=3,
        alpha=1,
        bagFraction=0.8,
        _lambda=0.1,
        gamma=0.1,
    ):
        # 追加する木の本数（ブースト回数）
        self.t_numbers = t_numbers
        # 各決定木の最大深さ
        self.depth_maximum = depth_maximum
        # 学習率に相当するスケール係数 α（負勾配のスケール）
        self.alpha = alpha
        # 各ステップでのサブサンプリング率（<1 なら Stochastic Gradient Boosting）
        self.bagFraction = bagFraction
        # 決定木側の L2 正則化係数 λ
        self._lambda = _lambda
        # 葉ノード数ペナルティ γ
        self.gamma = gamma

    def fit(self, inputs, target):
        """
        勾配ブースティングの学習アルゴリズム。

        1. 初期モデル F0(x) を 1 本目の木で学習
        2. 負の勾配（≒残差） g = α (y - Fm) を計算
        3. g をターゲットに新しい木を学習し，その出力を Fm に加算
        4. Fm を更新して再び残差を計算する

        これを t_numbers 回だけ繰り返し，関数 Fm を真の関数に近づけていく。
        """
        self.use_trees = []

        # ---------- ステップ 1: 初期モデル F0 ----------
        tree = DecisionTree(
            depth_maximum=self.depth_maximum, _lambda=self._lambda, gamma=self.gamma
        )
        # 元の目的変数 target に対して 1 本目の木を学習
        tree.fit(inputs, target)
        # 初期予測 F0(x)
        F0 = tree.predict(inputs)
        # 二乗誤差に対する負勾配 g = α (y - F0) （勾配方向）
        gradient = self.alpha * (target - F0)
        # 現在のモデル Fm を F0 で初期化
        Fm = F0
        # 1 本目の木をアンサンブルに登録
        self.use_trees.append(tree)

        # ---------- ステップ 2 以降: 残差を順次フィット ----------
        for i in range(self.t_numbers - 1):
            # bagFraction < 1 の場合は subsampling による Stochastic GB を行う
            if self.bagFraction < 1.0:
                # 使用するサンプル数
                baggings = int(round(inputs.shape[0] * self.bagFraction))
                # ランダムに baggings 個のインデックスを取得（置換なし）
                idx = np.random.choice(range(inputs.shape[0]), baggings, replace=False)
                x = inputs[idx]
                # サンプリングされたデータ上の残差をターゲットとして使う
                y = gradient[idx]
            else:
                # サブサンプリングしない場合は全データを使用
                x = inputs
                y = gradient

            # 残差 y を回帰する新しい決定木を学習（λ, γ を含む分割コストで構築）
            tree = DecisionTree(
                depth_maximum=self.depth_maximum, _lambda=self._lambda, gamma=self.gamma
            )
            tree.fit(x, y)

            # ---- Fm の更新（式(45) に対応）----
            # 新しい木が近似した残差をそのまま Fm に加算している。
            # 本家 XGBoost では learning_rate や葉重みの解析解が入るが，
            # ここでは簡略化された更新になっている点に注意。
            Fm += tree.predict(inputs)

            # ---- 新しい残差の計算（式(43) に対応）----
            # 更新後のモデルに対する負勾配 g = α (y - Fm) を再計算し，
            # 次の木のターゲットとする。
            gradient = self.alpha * (target - Fm)

            # 学習済みの木をアンサンブルに追加
            self.use_trees.append(tree)

    def predict(self, inputs):
        """
        学習済みモデルによる予測。

        ・各ステップで学習した木 f_m(x) の出力をすべて足し合わせることで
          最終的な予測 F_M(x) を得る（F0 + f1 + ... + f_{M-1}）。
        """
        predicts = [tree.predict(inputs) for tree in self.use_trees]
        return np.sum(predicts, axis=0)


def main():
    # -----------------------------
    # データ準備（単純な 1 次元回帰データ）
    # -----------------------------
    inputs = np.array([5.0, 7.0, 12.0, 20.0, 23.0, 25.0, 28.0, 29.0, 34.0, 35.0, 40.0])
    target = np.array(
        [62.0, 60.0, 83.0, 120.0, 158.0, 172.0, 167.0, 204.0, 189.0, 140.0, 166.0]
    )

    # XGBoost 風モデルを構築
    # t_numbers=5: 5 本の木を積み上げる
    # depth_maximum=3: 各木の深さを 3 に制限（弱学習器としてはほどよい）
    # _lambda=0, gamma=0 としているので，この設定では正則化はオフに近く，
    # 実質的には「単純な勾配ブースティング決定木」に近い挙動になる。
    plf = XGBoost(t_numbers=5, depth_maximum=3, _lambda=0, gamma=0)

    # モデルの学習
    plf.fit(inputs, target)

    # 学習データ上での予測
    y_pred = plf.predict(inputs)
    print(y_pred)

    # 観測データと予測値の可視化
    plt.scatter(inputs, target, label="data")  # 実データ
    plt.step(
        inputs, y_pred, color="orange", label="prediction"
    )  # XGBoost 風モデルの予測
    plt.ylim(10, 210)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
