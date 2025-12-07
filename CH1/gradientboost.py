import numpy as np
import matplotlib.pyplot as plt


class DecisionTree:
    def __init__(self, split_minimum=2, depth_maximum=100):
        # ノードを分割するために必要な最小サンプル数
        self.split_minimum = split_minimum
        # 木の最大深さ（大きくしすぎると過学習しやすくなる）
        self.depth_maximum = depth_maximum

    def tree_arrangement(self, inputs, node):
        """
        1 サンプルの入力値に対し，学習済み決定木から予測値を取り出す再帰関数。

        ・葉ノード（val が None でない）に到達したら，そのノードに格納された値を返す。
        ・内部ノードでは，しきい値 thrs と比較して，左の子ノード／右の子ノードへ降りていく。
        """
        if node.val is not None:
            # 葉ノード：格納してある回帰値を返す
            return node.val
        # しきい値以下なら左へ
        if inputs <= node.thrs:
            return self.tree_arrangement(inputs, node.left)
        # それ以外は右へ
        return self.tree_arrangement(inputs, node.right)

    def entropy(self, target):
        """
        target の経験分布に対するエントロピーを計算する。

        本来エントロピーは分類タスクでのクラス分布の不確実性を測る指標
            H(p) = - Σ p_k log2 p_k
        であり，ここでは連続値 target を「離散カテゴリ」とみなして，
        そのばらつき（不確実性）を擬似的に測っている。
        """
        # ユニーク値ごとの出現頻度をカウント
        _, hist = np.unique(target, return_counts=True)
        # 出現頻度を確率分布に正規化
        p = hist / len(target)
        # エントロピー H(p) = - Σ p log2 p
        return -np.sum(p * np.log2(p))

    def tree_growth(self, inputs, target, depth=0):
        """
        決定木を再帰的に成長させるメイン関数。

        ・現在のノードに属するサンプル (inputs, target) を最も良く 2 分割するしきい値を探索する。
        ・分割の良さは「情報利得（エントロピーの減少量）」で評価する。
        ・最大深さやサンプル数の制約を満たすと葉ノードを作り，target の平均値を予測値として持たせる。
        """
        # 現在のノードに含まれるサンプル数
        samples = inputs.shape[0]

        # 終了条件：最大深さを超えた，もしくはサンプル数が少なすぎる場合
        if depth >= self.depth_maximum or samples < self.split_minimum:
            # 回帰問題なので，このノードでは target の平均値を返すようにする
            return Tree_Node(val=np.mean(target))

        # 1 次元入力を前提に，取り得る全ての値をしきい値候補とする
        thresholds = np.unique(inputs)
        best_gain = -1

        # すべてのしきい値候補 th について情報利得を計算
        for th in thresholds:
            # 左ノード: x <= th, 右ノード: x > th
            idx_left = np.where(inputs <= th)
            idx_right = np.where(inputs > th)

            # どちらか一方にサンプルが無い場合は，実質的に分割できないので利得 0
            if len(idx_left[0]) == 0 or len(idx_right[0]) == 0:
                gain = 0
            else:
                # 親ノード（分割前）のエントロピー
                original_entropy = self.entropy(target)
                # 左右ノードのエントロピー
                e_left = self.entropy(target[idx_left])
                e_right = self.entropy(target[idx_right])
                # 左右ノードのサンプル数
                n_left, n_right = len(idx_left[0]), len(idx_right[0])
                # 分割後のエントロピー（重み付き平均）
                weighted_average_entropy = e_left * (n_left / samples) + e_right * (
                    n_right / samples
                )
                # 情報利得 = 分割前エントロピー − 分割後エントロピー
                gain = original_entropy - weighted_average_entropy

            # これまでで最も情報利得が大きい分割を保存
            if gain > best_gain:
                index_left = idx_left
                index_right = idx_right
                best_gain = gain
                threshhold_best = th

        # 情報利得が 0 の場合は，これ以上分割しても不確実性が減らないので葉ノードとする
        if best_gain == 0:
            return Tree_Node(val=np.mean(target))

        # 左右それぞれに対して再帰的に木を成長させる
        left_node = self.tree_growth(inputs[index_left], target[index_left], depth + 1)
        right_node = self.tree_growth(
            inputs[index_right], target[index_right], depth + 1
        )
        # 現在のノードは内部ノードとして，最良しきい値と子ノードへの参照を保持
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
        各入力値ごとに root から葉までたどり，葉ノードの val を返す。
        """
        return np.array(
            [self.tree_arrangement(input_, self.root_node) for input_ in inputs]
        )


class Tree_Node:
    """
    決定木の 1 つのノードを表すクラス。

    ・内部ノード: thrs に分割しきい値，left/right に子ノード，val は None
    ・葉ノード: val に予測値（回帰値）を格納し，thrs/left/right は None
    """

    def __init__(self, thrs=None, left=None, right=None, *, val=None):
        self.thrs = thrs
        self.left = left
        self.right = right
        self.val = val


class G_Boost:
    """
    Gradient Boosting（勾配ブースティング）による 1 次元回帰モデル。

    ・弱学習器として浅い決定木を順番に学習し，
      各ステップで「現在のモデルの残差（勾配）」を近似する木を追加していく。
    ・目的関数として二乗誤差
        L(F) = 1/2 Σ (y_i - F(x_i))^2
      を考えると，その F に関する負の勾配は (y_i - F(x_i)) となる。
      したがって，各ステップで残差 (y - Fm) を回帰する木を足していくことは，
      勾配降下法によって関数空間上で L(F) を最小化していることに対応する。
    """

    def __init__(self, t_numbers=5, depth_maximum=5, gamma=1, bagFraction=0.8):
        # 追加する決定木の本数（ブースト回数）
        self.t_numbers = t_numbers
        # 各決定木の最大深さ
        self.depth_maximum = depth_maximum
        # 学習率（shrinkage パラメータ）。ここでは勾配のスケール係数として使用
        self.gamma = gamma
        # 各ステップでデータの何割をサブサンプリングするか（<1であれば stochastic GB になる）
        self.bagFraction = bagFraction

    def fit(self, inputs, target):
        """
        勾配ブースティングの学習アルゴリズム。

        1. 初期モデル F0(x) を 1 本目の決定木で学習
        2. 残差（=二乗誤差に対する負勾配） g = γ (y - Fm) を計算
        3. g を目的値として新しい決定木を学習し，その出力を Fm に加算
        4. 新しい Fm に対して再び残差 g を計算し，これを繰り返す

        これにより，関数 Fm が段階的に真の関数に近づいていく。
        """
        self.use_trees = []

        # ---------- ステップ 1: 初期モデル F0(x) ----------
        # まず元のターゲット y に対して決定木を 1 本学習し，初期予測 F0(x) とする
        tree = DecisionTree(depth_maximum=self.depth_maximum)
        tree.fit(inputs, target)
        # 初期モデル F0(x) をデータ点上で計算
        F0 = tree.predict(inputs)
        # 二乗誤差の負勾配（残差）: g = γ (y - F0)
        # → 「まだ説明しきれていない分」を次の木で学習させる
        gradient = self.gamma * (target - F0)

        # アンサンブルに 1 本目の木を追加
        self.use_trees.append(tree)
        # 現在のモデル Fm を F0 で初期化
        Fm = F0

        # ---------- ステップ 2 以降: 残差を順次学習 ----------
        for _ in range(self.t_numbers - 1):
            # bagFraction < 1 の場合は subsampling による stochastic gradient boosting
            if self.bagFraction < 1.0:
                # 使用するサンプル数（全体の bagFraction 割）
                baggings = int(round(inputs.shape[0] * self.bagFraction))
                # ランダムに baggings 個のインデックスを取得（置換なし）
                idx = np.random.choice(range(inputs.shape[0]), baggings, replace=False)
                x = inputs[idx]
                # そのサンプルにおける残差（勾配）を目的値として使う
                y = gradient[idx]
            else:
                # サブサンプリングなしの場合は全データを使用
                x = inputs
                y = gradient

            # 残差 y を回帰する新しい決定木を学習
            tree = DecisionTree(depth_maximum=self.depth_maximum)
            tree.fit(x, y)

            # ---------- Fm の更新（式(34),(35) 相当） ----------
            # 新しい木の予測値を「勾配方向へのステップ」として加算
            # （gamma を木側に掛ける実装もあるが，ここでは前段で勾配に掛けている）
            Fm += tree.predict(inputs)

            # ---------- 新しい残差の計算（式(37) 相当） ----------
            # 更新後のモデル Fm に対する残差 g = γ (y - Fm)
            gradient = self.gamma * (target - Fm)

            # 学習済みの木をアンサンブルに追加
            self.use_trees.append(tree)

    def predict(self, inputs):
        """
        学習済みモデルによる予測。

        ・各ステップで学習した木 f_m(x) の出力をすべて足し合わせることで
          最終的な予測 F_M(x) を得る。
        """
        # すべての木の予測 f_m(x) を計算し，サンプルごとに和を取る
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

    # 勾配ブースティングモデルの構築
    # t_numbers=5: 5 本の木を段階的に追加
    # depth_maximum=2: 各木の深さを 2 に制限し，弱学習器として使用
    plf = G_Boost(t_numbers=5, depth_maximum=2)

    # モデルの学習
    plf.fit(inputs, target)

    # 学習データ上での予測
    y_pred = plf.predict(inputs)
    print(y_pred)

    # 観測データと予測値の可視化
    plt.scatter(inputs, target, label="data")  # 実データ
    plt.step(
        inputs, y_pred, color="orange", label="prediction"
    )  # 勾配ブースティングの予測
    plt.ylim(10, 210)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
