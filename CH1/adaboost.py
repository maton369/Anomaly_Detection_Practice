import numpy as np
import matplotlib.pyplot as plt


class DecisionTree:
    def __init__(self, split_minimum=2, depth_maximum=100):
        # 1 ノードを分割するために必要な最小サンプル数
        self.split_minimum = split_minimum
        # 木の最大深さ（大きくしすぎると過学習しやすくなる）
        self.depth_maximum = depth_maximum

    def tree_arrangement(self, inputs, node):
        """
        学習済み決定木から 1 サンプル分の予測値を取り出す再帰関数。

        ・葉ノード（val が None でない）に到達したら、その値を返す。
        ・内部ノードでは「入力値 <= しきい値」かどうかで左右どちらの子ノードへ進むか決める。
        """
        if node.val is not None:
            # 葉ノード：格納されている回帰値を返す
            return node.val
        # しきい値以下なら左の枝へ
        if inputs <= node.thrs:
            return self.tree_arrangement(inputs, node.left)
        # それ以外は右の枝へ
        return self.tree_arrangement(inputs, node.right)

    def entropy(self, target):
        """
        target の経験分布に対するエントロピーを計算する。

        本来エントロピーは分類タスクでのクラス分布の不確実性を測る指標
        $$H(p)=-\sum_k p_k\log_2 p_k$$
        であり、値が均等に散らばっているほど大きくなる。
        ここでは回帰値 target を離散カテゴリのように扱って
        その散らばり具合を擬似的に測っている。
        """
        # ユニーク値ごとの出現回数をカウント
        _, hist = np.unique(target, return_counts=True)
        # 出現頻度を確率に正規化
        p = hist / len(target)
        # H(p) = - Σ p log2 p
        return -np.sum(p * np.log2(p))

    def tree_growth(self, inputs, target, depth=0):
        """
        決定木を再帰的に成長させる。

        ・現在のノードに属するサンプル (inputs, target) から
          最も情報利得（エントロピー減少）が大きくなるしきい値を探索し、
          左右の子ノードに分割する。
        ・最大深さやサンプル数の条件を満たした場合や、
          これ以上情報利得が得られない場合は葉ノードを作成し、
          そのノードでは target の平均値を予測値として格納する（回帰タスク）。
        """
        samples = inputs.shape[0]

        # 深さ制限 or サンプル不足なら葉ノードで打ち切り
        if depth >= self.depth_maximum or samples < self.split_minimum:
            return Tree_Node(val=np.mean(target))

        # 1 次元入力を前提として、取り得るユニークな値をすべてしきい値候補とする
        thresholds = np.unique(inputs)
        best_gain = -1

        # 全てのしきい値候補について情報利得を計算
        for th in thresholds:
            # 左ノード：しきい値以下 / 右ノード：しきい値より大きい
            idx_left = np.where(inputs <= th)
            idx_right = np.where(inputs > th)

            # 片側にサンプルがない場合は分割しても意味がないので利得 0
            if len(idx_left[0]) == 0 or len(idx_right[0]) == 0:
                gain = 0
            else:
                # 親ノードのエントロピー
                original_entropy = self.entropy(target)
                # 左右ノードそれぞれのエントロピー
                e_left = self.entropy(target[idx_left])
                e_right = self.entropy(target[idx_right])
                # 左右ノードのサンプル数
                n_left, n_right = len(idx_left[0]), len(idx_right[0])
                # 分割後のエントロピー（重み付き平均）
                weighted_average_entropy = e_left * (n_left / samples) + e_right * (
                    n_right / samples
                )
                # 情報利得 = 分割前エントロピー - 分割後エントロピー
                gain = original_entropy - weighted_average_entropy

            # これまでで最も良い分割を記録
            if gain > best_gain:
                index_left = idx_left
                index_right = idx_right
                best_gain = gain
                threshhold_best = th

        # 情報利得が 0（=分割してもエントロピーが減らない）なら葉ノードにする
        if best_gain == 0:
            return Tree_Node(val=np.mean(target))

        # 左右それぞれの部分集合に対して再帰的に木を構築
        left_node = self.tree_growth(inputs[index_left], target[index_left], depth + 1)
        right_node = self.tree_growth(
            inputs[index_right], target[index_right], depth + 1
        )
        # 内部ノードには最良しきい値と左右の子ノードを持たせる
        return Tree_Node(threshhold_best, left_node, right_node)

    def fit(self, inputs, target):
        """
        決定木の学習インターフェース。

        1 本の木の根ノード（self.root_node）を構築する。
        """
        self.root_node = self.tree_growth(inputs, target)

    def predict(self, inputs):
        """
        学習済み決定木を使って予測を行う。

        各入力値に対して root から葉まで辿り、その葉の val を返す。
        """
        return np.array(
            [self.tree_arrangement(input_, self.root_node) for input_ in inputs]
        )


class Tree_Node:
    """
    決定木を構成するノードクラス。

    ・内部ノード: thrs にしきい値、left/right に子ノード、val は None
    ・葉ノード: val に予測値（回帰値）を格納し、thrs/left/right は None
    """

    def __init__(self, thrs=None, left=None, right=None, *, val=None):
        self.thrs = thrs
        self.left = left
        self.right = right
        self.val = val


class AdaBoost:
    """
    AdaBoost.R2 風の回帰用 AdaBoost の簡易実装。

    ・弱学習器として浅い決定木（DecisionTree）を複数本学習し、
      各木に誤差に基づく重み（重要度）を付けてアンサンブルする。
    ・学習時には「誤差が大きいサンプル」の重みを増やし、
      次の木がそれらを重点的に学習するようにするのが AdaBoost の基本アイデア。
    """

    def __init__(self, t_numbers=20, depth_maximum=5):
        # 学習する弱学習器（決定木）の本数
        self.t_numbers = t_numbers
        # 各決定木の最大深さ（弱学習器なので浅めにするのが典型的）
        self.depth_maximum = depth_maximum

    def fit(self, inputs, target):
        """
        AdaBoost による学習処理。

        1. サンプル重み w を一様に初期化
        2. 各ラウンド m で
           - w に基づく確率分布から bootstrap サンプリングして決定木を学習
           - その木の誤差率 em^bar を計算
           - em^bar から木の重み γ_m を計算
           - サンプルごとの損失に応じて w を更新
        という手続きを繰り返す。
        """
        self.use_trees = []  # 学習に採用された決定木のリスト
        self.gamma = np.zeros(self.t_numbers)  # 各木の「逆の強さ」に対応する係数 γ_m
        # サンプル重み w_i を一様に初期化（Σ w_i = 1）
        weights = np.ones(inputs.shape[0]) / inputs.shape[0]
        all_idx = np.arange(inputs.shape[0])

        for i in range(self.t_numbers):
            # ラウンド m における弱学習器（決定木）を生成
            tree = DecisionTree(depth_maximum=self.depth_maximum)

            # 現在の重みを確率分布に正規化
            # → w_i が大きいサンプルほど選ばれやすくなる
            avg_weight = weights / weights.sum()
            # この確率分布に従って bootstrap サンプリング（重複あり）
            idx = np.random.choice(
                all_idx, size=inputs.shape[0], replace=True, p=avg_weight
            )

            # サンプリングされたデータで決定木を学習
            tree.fit(inputs[idx], target[idx])
            # その木による予測値
            output = tree.predict(inputs[idx])

            # ==== 平均誤差率 em^bar の計算（AdaBoost.R2 の式(28) 相当）====
            # 絶対誤差
            error = abs(output - target[idx])
            # 誤差を 0〜1 に正規化（最大誤差で割る）
            # → Ei = 0: 完全に正解, Ei ≈ 1: かなり外している
            loss = error / (max(error) + 1e-50)
            # 現在の重み w_i を用いた損失の期待値 em^bar = Σ w_i Ei
            error_bar = np.sum(weights * loss)
            print(f"tree #{i+1} : error_bar = {error_bar}")

            # ==== 学習の打ち切り条件 ====
            # em^bar = 0: 全サンプルを完全に説明する強い木が得られた
            # em^bar >= 0.5: 弱学習器として不適（ランダム予測レベル以上に悪い）
            if error_bar == 0 or error_bar >= 0.5:
                if i == 0:
                    # 1 本目からすでにこの条件を満たす場合はその木だけ採用
                    self.use_trees.append(tree)
                    self.gamma = self.gamma[: i + 1]
                    break
                else:
                    # それ以外の場合は直前までの木のみを有効とする
                    self.gamma = self.gamma[:i]
                    break

            # 条件を満たさないのでこの木をアンサンブルに追加
            self.use_trees.append(tree)

            # ==== 各決定木に対応する係数 γ_m の計算（式(29)）====
            # AdaBoost.R2 では
            #   γ_m = em^bar / (1 - em^bar)
            # とし，予測時には log(1/γ_m) が弱学習器の重み α_m に対応する。
            self.gamma[i] = error_bar / (1.0 - error_bar)

            # ==== サンプル重み w の更新（式(30)）====
            # loss Ei が大きいサンプルほど (1 - Ei) が小さくなるので
            #   γ_m^{1 - Ei}
            # は大きくなり、結果として w_i が増加する。
            # → 誤差の大きいサンプルの重みが強調され、次の木がそれらを重視して学習する。
            weights *= [np.power(self.gamma[i], 1.0 - Ei) for Ei in loss]
            # 正規化して確率分布に戻す（Σ w_i = 1）
            weights /= weights.sum()

    def predict(self, inputs):
        """
        学習済み AdaBoost モデルによる予測。

        ・各弱学習器の予測を集め、
        ・γ_m から計算される重み α_m = log(1/γ_m) に基づく「重み付き中央値」を返す。

        AdaBoost.R2 では、重み付き平均ではなく重み付き中央値を用いることが提案されており、
        これは外れ値に対するロバスト性を高めるための工夫である。
        """
        # 各決定木の予測値（shape: [n_trees, n_samples]）
        predicts = np.array([tree.predict(inputs) for tree in self.use_trees])

        # 木が 1 本しかない場合はそのまま返す
        if len(self.use_trees) == 1:
            return predicts[0]
        else:
            # γ_m から α_m = log(1/γ_m) を計算
            # これは分類版 AdaBoost の α_m に対応する重み
            theta = np.log(1.0 / self.gamma)

            # 各サンプルごとに予測値の昇順に並べ替え、その順序 idx を取得
            idx = np.argsort(predicts, axis=0)

            # 並べ替え後の順に重みを累積（重み付き CDF）
            cdf = theta[idx].cumsum(axis=0)

            # CDF が全体重みの半分を初めて超える位置が「重み付き中央値」
            above = cdf >= theta.sum() / 2
            median_idx = above.argmax(axis=0)

            # 各サンプルについて「中央値に対応する弱学習器のインデックス」を取り出す
            median_estimators = np.diag(idx[median_idx])

            # その弱学習器の予測を最終予測として返す
            return np.diag(predicts[median_estimators])


def main():
    # データ準備（単純な 1 次元回帰データ）
    inputs = np.array([5.0, 7.0, 12.0, 20.0, 23.0, 25.0, 28.0, 29.0, 34.0, 35.0, 40.0])
    target = np.array(
        [62.0, 60.0, 83.0, 120.0, 158.0, 172.0, 167.0, 204.0, 189.0, 140.0, 166.0]
    )

    # AdaBoost 回帰モデルを構築
    # t_numbers=10: 最大 10 本まで弱学習器を追加
    # depth_maximum=3: 各決定木の深さを 3 に制限（弱学習器として適度な表現力）
    plf = AdaBoost(t_numbers=10, depth_maximum=3)

    # モデルの学習
    plf.fit(inputs, target)

    # 学習データに対する予測
    y_pred = plf.predict(inputs)
    print(y_pred)

    # 観測値と予測値の可視化
    plt.scatter(inputs, target, label="data")  # 実データ
    plt.step(
        inputs, y_pred, color="orange", label="prediction"
    )  # AdaBoost の予測（階段状）
    plt.ylim(10, 210)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
