import numpy as np
import matplotlib.pyplot as plt


class DecisionTree:
    def __init__(self, split_minimum=2, depth_maximum=100):
        # ノードを分割するために必要なサンプル数の最小値
        # → サンプル数がこれ未満のノードはそれ以上分割しない（過学習抑制＋葉の安定性確保）
        self.split_minimum = split_minimum
        # 木の最大深さ
        # → 深さ制限を設けることで、極端に複雑な木になることを防ぐ
        self.depth_maximum = depth_maximum

    def tree_arrangement(self, inputs, node):
        """
        1 つの入力値に対して、既に構築された決定木から予測値を取り出す再帰関数。

        ・葉ノードに到達したら、そのノードに格納してある予測値（val）を返す。
        ・内部ノードでは、しきい値 thrs と比較して左右どちらの子ノードに降りるかを決める。
        """
        # val が None でなければ葉ノードであり、そこに格納された予測値を返す
        if node.val is not None:
            return node.val
        # 現在のノードのしきい値より小さい（または等しい）場合は左の枝へ
        if inputs <= node.thrs:
            return self.tree_arrangement(inputs, node.left)
        # それ以外は右の枝へ
        return self.tree_arrangement(inputs, node.right)

    def entropy(self, target):
        """
        ターゲット値のエントロピーを計算する関数。

        本来エントロピーは分類におけるクラス分布の不確実性を測る指標で、
        確率分布 p に対して

            H(p) = - Σ_k p_k log2 p_k

        と定義される。

        ここでは target 内の値の相対頻度を「確率」とみなし、
        そのエントロピーを計算している。
        （回帰タスクにエントロピーを使うのは少し特殊だが、
         「値の散らばり具合」を擬似的に測る目的で使っていると解釈できる。）
        """
        # target 内のユニーク値と、それぞれの出現回数を取得
        _, hist = np.unique(target, return_counts=True)
        # 出現回数を全体で割って「経験的な確率分布」に変換
        p = hist / len(target)
        # エントロピー H = - Σ p log2 p
        return -np.sum(p * np.log2(p))

    def tree_growth(self, inputs, target, depth=0):
        """
        決定木を再帰的に成長させる関数。

        ・現在のノードに対応するサンプル集合 (inputs, target) を受け取り、
          最適なしきい値で 2 分割することで左右の子ノードを生成する。
        ・分割の良さは「情報利得（エントロピーの減少量）」で評価する。

            gain = H(parent) - [ (n_left / n) H(left) + (n_right / n) H(right) ]

        ・終了条件（最大深さ・サンプル数不足・情報利得 0）を満たした場合は
          葉ノードを作り、そのノードには target の平均値を格納する（回帰なので平均予測）。
        """
        # 現在のノードに含まれるサンプル数
        samples = inputs.shape[0]

        # 終了条件1: 最大深さを超えたら葉ノードにする
        # 終了条件2: サンプル数が split_minimum 未満なら葉ノードにする
        if depth >= self.depth_maximum or samples < self.split_minimum:
            # 回帰木なので、このノードでは target の平均値を予測値として持たせる
            return Tree_Node(val=np.mean(target))

        # 1 次元入力を前提として、取り得るすべての値を候補しきい値とする
        thresholds = np.unique(inputs)

        # これまで見つかった中での最大情報利得
        best_gain = -1

        # すべてのしきい値候補について、「どこで分割するとエントロピーが一番減るか」を探索
        for th in thresholds:
            # 左ノード: しきい値以下
            idx_left = np.where(inputs <= th)
            # 右ノード: しきい値より大きい
            idx_right = np.where(inputs > th)

            # 片側にサンプルがなくなる（=実質的に分割できない）場合は利得 0
            if len(idx_left[0]) == 0 or len(idx_right[0]) == 0:
                gain = 0
            else:
                # --- 分割基準の計算 ---
                # 下のコメントのように、分類問題の場合にはジニ係数を使うのが一般的。
                # ここでは回帰タスクだが、エントロピーを用いた「情報利得」という
                # 分割基準を採用している。

                # （分類問題での例：ジニ係数）
                # p1_node1, p2_node1 = probability(target[idx_left])
                # p1_node2, p2_node2 = probability(target[idx_right])
                # sample_sum_node1, sample_sum_node2 = len(idx_left), len(idx_right)
                # gini_node1 = 1 - p1_node1**2 - p2_node1**2
                # gini_node2 = 1 - p1_node2**2 - p2_node2**2
                # gain = gini_node1*(sample_sum_node1 / samples) \
                #     + gini_node2 * (sample_sum_node2 / samples)

                # --- 回帰問題としての分割基準 ---
                # 「分割前のエントロピー」と「分割後のエントロピーの重み付き平均」を比べる。
                # 情報利得 = H(親ノード) - [ (n_left / n) H(左) + (n_right / n) H(右) ]
                original_entropy = self.entropy(target)
                e_left = self.entropy(target[idx_left])
                e_right = self.entropy(target[idx_right])
                # 左右ノードのサンプル数
                n_left, n_right = len(idx_left[0]), len(idx_right[0])
                # 分割後のエントロピー（重み付き平均）
                weighted_average_entropy = e_left * (n_left / samples) + e_right * (
                    n_right / samples
                )
                # 情報利得（エントロピーの減少量）
                gain = original_entropy - weighted_average_entropy

            # これまでで最大の情報利得を更新した場合、そのときの分割を採用候補として保存
            if gain > best_gain:
                index_left = idx_left
                index_right = idx_right
                best_gain = gain
                threshhold_best = th

        # 情報利得が 0（=分割によってエントロピーが減らない）なら、これ以上分割する意味がない
        if best_gain == 0:
            # 葉ノードにして、ここでも target の平均値を予測値として格納
            return Tree_Node(val=np.mean(target))

        # 左右それぞれのサブセットに対して、再帰的に木を成長させる
        left_node = self.tree_growth(inputs[index_left], target[index_left], depth + 1)
        right_node = self.tree_growth(
            inputs[index_right], target[index_right], depth + 1
        )

        # 現在のノードには最良しきい値を持たせ、左右の子ノードを接続
        return Tree_Node(threshhold_best, left_node, right_node)

    def fit(self, inputs, target):
        """
        決定木の学習を行うインターフェース。

        ここでは 1 本の木を根ノード self.root_node として構築する。
        """
        self.root_node = self.tree_growth(inputs, target)

    def predict(self, inputs):
        """
        学習済みの決定木を使って、各入力に対する予測値を返す。

        各入力点に対して root から葉まで辿ることで予測を行う。
        """
        return np.array(
            [self.tree_arrangement(input_, self.root_node) for input_ in inputs]
        )


class Tree_Node:
    """
    決定木の 1 ノードを表すクラス。

    ・内部ノード: thrs に分割しきい値、left/right に子ノード、val は None
    ・葉ノード: val に予測値を持ち、thrs/left/right は None
    """

    def __init__(self, thrs=None, left=None, right=None, *, val=None):
        self.thrs = thrs  # 分割しきい値（内部ノードで使用）
        self.left = left  # 左の子ノード
        self.right = right  # 右の子ノード
        self.val = val  # 葉ノードに格納される予測値（回帰なので実数）


class RandomForest:
    """
    単純な 1 次元用ランダムフォレスト回帰器。

    ・複数本の決定木を bootstrap サンプリングしたデータで学習し、
      予測時にはそれらの平均を取ることでバリアンス（分散）を下げる。

    理論的には、個々の木は高バリアンスな「弱学習器」だが、
    それらを多数平均することで

        ・過学習のリスクを減らし
        ・ノイズに対するロバスト性を高める

    という効果が期待できる。
    """

    def __init__(self, t_numbers=10, split_minimum=5, depth_maximum=100):
        # 森に含める決定木の本数
        self.t_numbers = t_numbers
        # 各決定木に渡すハイパーパラメータ
        self.split_minimum = split_minimum
        self.depth_maximum = depth_maximum

    def fit(self, inputs, target, node_num=10):
        """
        ランダムフォレストの学習。

        ・t_numbers 本の決定木を用意し、それぞれに対して
          bootstrap サンプリング（重複あり抽出）したサブセットで学習させる。
        ・これにより、各木が少しずつ異なる視点でデータを見て学習するため、
          アンサンブルとしての多様性が高まり、汎化性能が向上する。
        """
        self.use_trees = []
        for _ in range(self.t_numbers):
            # 個々の決定木を生成
            tree = DecisionTree(self.split_minimum, self.depth_maximum)
            # bootstrap サンプリングしたサブセットを取得
            x_samp, y_samp = self.sampling_bootstrap(inputs, target, node_num)
            # サブセットで決定木を学習
            tree.fit(x_samp, y_samp)
            # 学習済み木を森に追加
            self.use_trees.append(tree)

    def predict(self, inputs):
        """
        ランダムフォレストによる予測。

        ・各決定木の予測値を計算し、
        ・それらの平均を取ることで最終的な予測値とする。

        これは「バギング（Bootstrap Aggregating）」と呼ばれるアンサンブル手法で、
        各木のばらつきをならして安定した予測を得る狙いがある。
        """
        # 各木に対して predict を実行（shape: [n_trees, n_samples]）
        predicts = np.array([tree.predict(inputs) for tree in self.use_trees])
        # 木ごとの予測を平均して最終予測にする
        return np.mean(predicts, axis=0)

    def sampling_bootstrap(self, inputs, target, node_num):
        """
        bootstrap サンプリングを行う関数。

        ・元データから重複を許して node_num 個のサンプルをランダムに抽出する。
        ・同じインデックスが複数回選ばれることで「擬似的な再標本化」となり、
          個々の木が少しずつ異なる学習セットを持つようになる。
        """
        idx = np.random.choice(inputs.shape[0], node_num, replace=True)
        return inputs[idx], target[idx]


def main():
    # ===============================
    # データの準備（単純な 1 次元回帰データ）
    # ===============================
    inputs = np.array([5.0, 7.0, 12.0, 20.0, 23.0, 25.0, 28.0, 29.0, 34.0, 35.0, 40.0])
    target = np.array(
        [62.0, 60.0, 83.0, 120.0, 158.0, 172.0, 167.0, 204.0, 189.0, 140.0, 166.0]
    )

    # ランダムフォレスト回帰器を構築
    # ・t_numbers=3: 3 本の決定木からなる小さな森
    # ・depth_maximum=2: 各木の最大深さを 2 に制限（かなり浅い木）
    plf = RandomForest(t_numbers=3, depth_maximum=2)

    # モデルの学習
    plf.fit(inputs, target)

    # 学習データに対する予測値を計算
    y_pred = plf.predict(inputs)
    print(y_pred)

    # 観測データと予測値の可視化
    plt.scatter(inputs, target, label="data")  # 実データ
    plt.step(
        inputs, y_pred, color="orange", label="prediction"
    )  # ランダムフォレストの予測（階段状）
    plt.ylim(10, 210)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
