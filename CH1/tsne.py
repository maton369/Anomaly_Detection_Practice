import numpy as np
import matplotlib.pyplot as plt


class Tsne:
    def __init__(
        self,
        data,
        label,
        TSNE=True,
        learning_rate=0.01,
        MOMENTUM=0.9,
        iter=100,
        seed=123,
    ):
        # TSNE=True のときは通常の t-SNE（低次元側で t 分布）を使う
        # False のときは SNE（低次元側もガウス分布）に相当する分岐になっている
        self.TSNE = TSNE
        self.iteration = iter  # 勾配降下の反復回数
        self.label = label  # 可視化用ラベル
        self.X = data  # 高次元データ X（N×D 行列を想定）
        self.learning_rate = learning_rate
        self.momentum = MOMENTUM  # モメンタム項の係数
        self.seed = seed

    def connect_p(self, X, p_information=2):
        """
        高次元空間における「類似度」行列 P を構成する。

        - 各点 i と j のユークリッド距離からガウスカーネルに基づく条件付き確率
          $$p_{j|i}$$ を計算し，シンメトライズ
          $$P_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$
          を作る処理に対応する。
        - p_information は「perplexity」と同値であり，
          $$\text{Perplexity}(P_i) = 2^{H(P_i)}$$ がこの値になるように
          各点 i の分散 $$\sigma_i$$ を探索する。
        """
        # 距離の 2 乗ノルムを利用した距離行列の構築
        input_sum = np.sum(np.square(X), 1)
        neg_distance = np.add(np.add(-2 * np.dot(X, X.T), input_sum).T, input_sum)
        distance = -neg_distance  # distance[i,j] = ||x_i - x_j||^2

        # 各点ごとに分散 σ_i を二分探索で最適化（perplexity を合わせる）
        total = self.sigma_opt(distance, p_information)

        # 条件付き確率行列 P_{j|i} をガウス分布に基づいて計算（式(76)）
        _P = self.matrix_4_p(distance, total)

        # SNE / t-SNE の論文と同様に，対称化して最終的な P を作成
        P = (_P + _P.T) / (2.0 * _P.shape[0])
        return P

    # 式(76)の計算：距離と σ から条件付き確率 p_{j|i} を求める
    def matrix_4_p(self, distance, total=None, idx_zero=None):
        if total is not None:
            # total: 各サンプル i の σ_i
            # ガウス分布の分散に 2σ_i^2 が出てくるので q_sigma = 2σ_i^2
            q_sigma = 2.0 * np.square(total.reshape((-1, 1)))
            # 行ごとにスケールが異なる softmax を計算
            # $$p_{j|i} \propto \exp\left(-\frac{||x_i - x_j||^2}{2\sigma_i^2}\right)$$
            return self.cal_softmax(distance / q_sigma, idx_zero=idx_zero)
        else:
            # σ を固定した単純な softmax（理論的には上の式の特別な場合）
            return self.cal_softmax(distance, idx_zero=idx_zero)

    # 式(78)の計算：低次元空間でのガウスカーネル版 q_{ij}
    def connect_q(self, y):
        """
        低次元埋め込み Y に対して，ガウス分布に基づく類似度 Q を計算する。

        本来の t-SNE では t 分布を用いるが，
        こちらは SNE タイプ（ガウス）の定義に対応している。
        """
        input_sum = np.sum(np.square(y), 1)
        neg_distance = np.add(np.add(-2 * np.dot(y, y.T), input_sum).T, input_sum)
        distance = -neg_distance
        power_distance = np.exp(distance)
        np.fill_diagonal(power_distance, 0.0)  # self-similarity は 0 にする
        # $$q_{ij} = \frac{\exp(-||y_i-y_j||^2)}{\sum_{k\neq l}\exp(-||y_k-y_l||^2)}$$
        return power_distance / np.sum(power_distance), None

    # 式(78)の計算：t-分布（t-SNE 本来の定義）
    def tene_4_q(self, y):
        """
        t-SNE のコアとなる低次元側の相互類似度 Q を定義する。

        - 距離に対して $$q_{ij} \propto (1 + ||y_i - y_j||^2)^{-1}$$
          となる重みを用い，最後に正規化して分布 Q を得る。
        - ガウスに比べて裾が重い（heavy-tailed）ため「crowding problem」を緩和し，
          クラス間の疎な距離をより強調する効果がある。
        """
        input_sum = np.sum(np.square(y), 1)
        neg_distance = np.add(np.add(-2 * np.dot(y, y.T), input_sum).T, input_sum)
        distance = -neg_distance  # ||y_i-y_j||^2

        # t 分布に対応するカーネル
        # $$w_{ij} = (1 + ||y_i - y_j||^2)^{-1}$$
        inverse_distance = np.power(1.0 - distance, -1)
        np.fill_diagonal(inverse_distance, 0.0)
        # 上式を正規化して $$q_{ij}$$ を得る
        return inverse_distance / np.sum(inverse_distance), inverse_distance

    def cal_softmax(self, X, d_zero=True, idx_zero=None):
        """
        各行ごとに softmax を計算するユーティリティ。

        - 数値安定化のために max を引いてから指数をとる（log-sum-exp trick）。
        - 対角要素（自己との距離）はゼロにすることで self-similarity を除去。
        """
        exp_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))
        if idx_zero is None:
            if d_zero:
                np.fill_diagonal(exp_x, 0.0)
        else:
            exp_x[:, idx_zero] = 0.0
        exp_x = exp_x + 1e-8  # ゼロ割防止のための微小値
        # 行ごとに正規化して確率分布にする
        return exp_x / exp_x.sum(axis=1).reshape([-1, 1])

    def _grad_now(self, Q, Y, distance):
        """
        t-SNE の目的関数である KL ダイバージェンス

        $$\mathrm{KL}(P\|Q) = \sum_{i\neq j} P_{ij}\log\frac{P_{ij}}{Q_{ij}}$$

        の勾配に相当する項を計算する（t-SNE 版）。
        """
        # y_i - y_j を全ペアについて並べたテンソル (N,N,2)
        y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)

        # t 分布版では勾配に (1 + ||y_i-y_j||^2)^{-1} が係数として掛かる
        dist_expanded = np.expand_dims(distance, 2)
        y_diffs_wt = y_diffs * dist_expanded

        # P - Q の行列（「高次元での類似度」と「低次元での類似度」の差）
        pq_diff = self.connect_p(self.X) - Q
        pq_expanded = np.expand_dims(pq_diff, 2)

        # まとめて
        # $$ \frac{\partial C}{\partial y_i}
        #    = 4 \sum_j (P_{ij}-Q_{ij}) (1+||y_i-y_j||^2)^{-1} (y_i-y_j) $$
        grad_now = 4.0 * (pq_expanded * y_diffs_wt).sum(1)
        return grad_now

    def _infos(self, p_matrix):
        """
        各点 i に対する条件付き確率分布 $$P_i$$ の情報量（エントロピー）を計算する。

        $$H(P_i) = -\sum_j P_{j|i}\log_2 P_{j|i}$$

        からパープレキシティ

        $$\text{Perplexity}(P_i) = 2^{H(P_i)}$$

        を得るため，その指数部のみを返している。
        """
        p_information = 2 ** -np.sum(p_matrix * np.log2(p_matrix), 1)
        return p_information

    def _information(self, distance, total, idx_zero):
        # sigma（分散）候補 total を与えたときのパープレキシティを計算するヘルパー
        return self._infos(self.matrix_4_p(distance, total, idx_zero))

    def sigma_opt(self, distance, target):
        """
        各データ点 i について，パープレキシティが target になるような
        ガウス分布の分散 $$\sigma_i$$ を二分探索で求める。

        - t-SNE では「各点が見る近傍の有効な数」を制御するハイパーパラメータが
          perplexity であり，この関数はそれを満たすように σ を調整している。
        """
        total = []
        for i in range(distance.shape[0]):

            def eval_fn(sigma):
                return self._information(distance[i : i + 1, :], np.array(sigma), i)

            correct_sigma = self._search(eval_fn, target)
            total.append(correct_sigma)
        return np.array(total)

    def tsne(self):
        """
        t-SNE / SNE の勾配降下アルゴリズム本体。

        - 高次元での類似度分布 P
        - 低次元での類似度分布 Q
        間の KL ダイバージェンスを最小化するように Y を更新する。
        """
        # 使用する勾配関数と Q の定義を t-SNE / SNE で切り替え
        self.grad_now_fn = self._grad_now if self.TSNE else self.grad_now_sym
        self.q_fn = self.tene_4_q if self.TSNE else self.connect_q

        rng = np.random.RandomState(self.seed)
        # 初期埋め込み Y を小さなガウスノイズから開始
        Y = rng.normal(0.0, 0.0001, [self.X.shape[0], 2])

        # モメンタム法のために前々回・前回の位置を保持
        if self.momentum:
            Y_m2 = Y.copy()
            Y_m1 = Y.copy()

        for _ in range(self.iteration):
            # 現在の Y から低次元分布 Q を計算
            Q, distance = self.q_fn(Y)

            # KL ダイバージェンスに対する勾配
            grad_nows = self.grad_now_fn(Q, Y, distance)

            # 勾配降下ステップ（単純な SGD）
            Y = Y - self.learning_rate * grad_nows

            # モメンタム項を付加して「慣性」を持たせる
            if self.momentum:
                Y += self.momentum * (Y_m1 - Y_m2)
                Y_m2 = Y_m1.copy()
                Y_m1 = Y.copy()

        return Y

    def _search(
        self, eval_fn, target, tol=1e-10, max_iter=10000, lower=1e-20, upper=1000.0
    ):
        """
        perplexity(target) に対応する σ を探すための二分探索。

        - eval_fn(σ) が実際の perplexity を返し，
          それが target に近づくように lower, upper の範囲を絞り込む。
        """
        for _ in range(max_iter):
            guess = (lower + upper) / 2.0
            val = eval_fn(guess)
            if val > target:
                upper = guess
            else:
                lower = guess
            if np.abs(val - target) <= tol:
                break
        # 元コードでは return がループ内にあり 1 ステップで終了していたので，
        # 理論どおりの二分探索になるようにループ後に返す形にするとよい。
        return guess

    def grad_now_sym(self, P, Q, Y):
        """
        SNE（ガウス分布版）での勾配計算。

        t-SNE では距離に応じた重みを追加しているのに対し，
        こちらは単純な

        $$ \frac{\partial C}{\partial y_i}
           = 4 \sum_j (P_{ij}-Q_{ij})(y_i-y_j) $$

        に相当する。
        """
        y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        pq_diff = P - Q
        pq_expanded = np.expand_dims(pq_diff, 2)
        grad_now = 4.0 * (pq_expanded * y_diffs).sum(1)
        return grad_now

    def plt_tsne(self, plt_data, title="", ms=6, ax=None, alpha=1.0, legend=True):
        """
        低次元埋め込み結果をラベルごとに色分けして描画するユーティリティ。
        """
        target = list(np.unique(self.label))
        write = "os" * len(target)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(target)))
        for i, cls in enumerate(target):
            mark = write[i]
            ax.plot(
                np.zeros(1),
                plt_data[self.label == cls],
                marker=mark,
                linestyle="",
                ms=ms,
                label=str(cls),
                alpha=alpha,
                color=colors[i],
                markeredgecolor="black",
                markeredgewidth=0.4,
            )
        if legend:
            ax.legend()
        ax.title.set_text(title)
        return ax

    def plt_input(self, plt_data, title="", ms=6, ax=None, alpha=1.0, legend=True):
        """
        元の高次元（ここでは 2 次元）データをラベルごとに描画するユーティリティ。
        """
        target = list(np.unique(self.label))
        write = "os" * len(self.label)
        for i, cls in enumerate(target):
            mark = write[i]
            ax.plot(
                plt_data[cls][0],
                plt_data[cls][1],
                marker=mark,
                linestyle="",
                ms=ms,
                label=str(cls),
                alpha=alpha,
                markeredgecolor="black",
                markeredgewidth=0.4,
            )
        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 12)
        ax.title.set_text(title)
        if legend:
            ax.legend()
        return ax


def main():
    # おもちゃデータ（2 次元）を作成
    x = [2.1, 1.1, 8.1, 0.9, 1.5, 1.5, 9.4, 1.3]
    y = [8.1, 7.5, 2.8, 1.7, 8.5, 1.8, 2.8, 2.2]
    inputs = np.array((x, y)).T
    label = np.arange(inputs.shape[0])

    # t-SNE による 2 次元埋め込み（ここでは元々 2 次元だが，挙動確認用）
    TSNE = Tsne(inputs, label, TSNE=True, seed=123)
    results = TSNE.tsne()

    # 元データと埋め込み結果を並べて描画
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    TSNE.plt_tsne(results, title="Tsne result", ax=ax[1])
    TSNE.plt_input(inputs, title="Data set", ax=ax[0])
    plt.show()


if __name__ == "__main__":
    main()
