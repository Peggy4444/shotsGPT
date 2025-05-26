import numpy as np
import pandas as pd
import math
from scipy.special import betaln
from graphviz import Digraph
import json


class Node:
    def __init__(self, depth=0):
        self.depth = depth
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.n = 0
        self.n_pos = 0.0
        self.prob = None

def beta_bernoulli_log_marginal(n_pos, n_neg, alpha=1.0, beta=1.0):
    return betaln(n_pos + alpha, n_neg + beta) - betaln(alpha, beta)

class BayesianClassificationTree:
    def __init__(self, alpha=2.0, beta=1.0, split_prior_decay=0.9,
                 min_samples=10, max_depth=4, split_precision=1e-6):
        self.alpha = alpha
        self.beta = beta
        self.split_prior_decay = split_prior_decay
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.split_precision = split_precision
        self.root = None
        self.feature_names = []

    def fit(self, X_df, y):
        self.feature_names = X_df.columns.tolist()
        X = X_df.values.astype(float)
        y_arr = y.values.astype(float)
        self.root = self._split_node(X, y_arr, depth=0)
        return self

    def _split_node(self, X, y, depth):
        node = Node(depth)
        n = y.shape[0]
        if n == 0:
            return node
        n_pos = y.sum()
        n_neg = n - n_pos
        node.n = n
        node.n_pos = n_pos
        node.prob = (n_pos + self.alpha) / (n + self.alpha + self.beta)

        trivial_score = beta_bernoulli_log_marginal(n_pos, n_neg, self.alpha, self.beta) + depth * math.log(self.split_prior_decay)
        best_score = trivial_score
        best_dim, best_thr, best_idxs = None, None, None

        if (self.max_depth and depth >= self.max_depth) or n < 2*self.min_samples:
            return node

        for j in range(X.shape[1]):
            col = X[:, j]
            sort_idx = np.argsort(col)
            col_sorted = col[sort_idx]
            diffs = np.abs(np.diff(col_sorted)) > self.split_precision
            split_positions = np.where(diffs)[0] + 1
            if split_positions.size == 0:
                continue

            y_sorted = y[sort_idx]
            cumsum_pos = np.cumsum(y_sorted)
            total_pos = n_pos

            for pos in split_positions:
                left_n, right_n = pos, n - pos
                if left_n < self.min_samples or right_n < self.min_samples:
                    continue
                left_pos = cumsum_pos[pos-1]
                right_pos = total_pos - left_pos
                left_neg = left_n - left_pos
                right_neg = right_n - right_pos
                score = beta_bernoulli_log_marginal(left_pos, left_neg, self.alpha, self.beta) + \
                        beta_bernoulli_log_marginal(right_pos, right_neg, self.alpha, self.beta) + \
                        depth * math.log(self.split_prior_decay)
                if score > best_score:
                    best_score = score
                    best_dim = j
                    best_thr = 0.5 * (col_sorted[pos-1] + col_sorted[pos])
                    best_idxs = (sort_idx[:pos], sort_idx[pos:])

        if best_dim is not None:
            node.feature = self.feature_names[best_dim]
            node.threshold = best_thr
            left_idx, right_idx = best_idxs
            node.left = self._split_node(X[left_idx], y[left_idx], depth+1)
            node.right = self._split_node(X[right_idx], y[right_idx], depth+1)
        return node

    def predict_proba(self, X_df):
        X = X_df[self.feature_names].values.astype(float)
        probs = []
        for row in X:
            node = self.root
            while node.feature is not None:
                idx = self.feature_names.index(node.feature)
                node = node.left if row[idx] <= node.threshold else node.right
            probs.append(node.prob)
        return np.array(probs)

    def path_contributions(self, X_df):
        X = X_df[self.feature_names].values.astype(float)
        contrib_list = []
        root_mean = self.root.prob
        for row in X:
            node = self.root
            prev = root_mean
            contribs = {feat: 0.0 for feat in self.feature_names}
            while node.feature is not None:
                idx = self.feature_names.index(node.feature)
                next_node = node.left if row[idx] <= node.threshold else node.right
                curr = node.prob
                contribs[node.feature] += (curr - prev)
                prev = curr
                node = next_node
            contrib_list.append(contribs)
        return pd.DataFrame(contrib_list, index=X_df.index)
    

        
    def to_dict(self) -> dict:
        """
        Walks the fitted tree and returns a nested dict of only
        Python primitives (no code objects).  This can be JSON-dumped.
        """
        def _node_to_dict(node):
            if node is None:
                return None

            return {
                "feature":   node.feature,
                "threshold": node.threshold,
                "n":         node.n,
                "n_pos":     node.n_pos,
                "prob":      node.prob,
                "left":      _node_to_dict(node.left),
                "right":     _node_to_dict(node.right),
            }

        return {
            "params": {
                "alpha":              self.alpha,
                "beta":               self.beta,
                "split_prior_decay":  self.split_prior_decay,
                "min_samples":        self.min_samples,
                "max_depth":          self.max_depth,
                "split_precision":    self.split_precision,
                "feature_names":      self.feature_names,
            },
            "root": _node_to_dict(self.root),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BayesianClassificationTree":
        """
        Reconstructs a BayesianClassificationTree from the dict
        produced by to_dict().
        """
        p = d["params"]
        # create a new instance with the same hyper-parameters
        inst = cls(
            alpha=p["alpha"],
            beta=p["beta"],
            split_prior_decay=p["split_prior_decay"],
            min_samples=p["min_samples"],
            max_depth=p["max_depth"],
            split_precision=p["split_precision"],
        )
        inst.feature_names = p["feature_names"]

        def _dict_to_node(nd):
            if nd is None:
                return None
            node = Node(depth=0)
            node.feature   = nd["feature"]
            node.threshold = nd["threshold"]
            node.n         = nd["n"]
            node.n_pos     = nd["n_pos"]
            node.prob      = nd["prob"]
            node.left      = _dict_to_node(nd["left"])
            node.right     = _dict_to_node(nd["right"])
            return node

        inst.root = _dict_to_node(d["root"])
        return inst

    
    # def to_graphviz(self):
    #     dot = Digraph(node_attr={"shape": "box", "fontsize": "10"})
    #     def recurse(node, name):
    #         #  create a label
    #         if node.feature is None:
    #             label = f"Leaf\nn={node.n}\np={node.prob:.2f}"
    #         else:
    #             label = (
    #                 f"{node.feature} ≤ {node.threshold:.2f}\n"
    #                 f"n={node.n}  p={node.prob:.2f}"
    #             )
    #         dot.node(name, label)


    #         if node.left:
    #             left_name = f"{name}L"
    #             recurse(node.left, left_name)
    #             dot.edge(name, left_name, label="True")
    #         if node.right:
    #             right_name = f"{name}R"
    #             recurse(node.right, right_name)
    #             dot.edge(name, right_name, label="False")

    #     recurse(self.root, "root")
    #     return dot
    def to_graphviz_with_path(self, X_row):
    

        dot = Digraph(node_attr={"shape": "box", "fontsize": "10"})
        path = []

        def get_path(node, row):
            path.append(id(node))
            if node.feature is None:
             return
            idx = self.feature_names.index(node.feature)
            if row[idx] <= node.threshold:
                get_path(node.left, row)
            else:
                get_path(node.right, row)

        row = X_row[self.feature_names].values.astype(float).flatten()
        get_path(self.root, row)

        def render_node(node, name):
            label = (
                f"{node.feature} ≤ {node.threshold:.2f}\n"
                f"n={node.n}  p={node.prob:.2f}"
             if node.feature is not None
                else f"Leaf\nn={node.n}\np={node.prob:.2f}"
         )
            is_path = id(node) in path
            style = "filled" if is_path else ""
            fillcolor = "lightblue" if is_path else "white"
            dot.node(name, label, style=style, fillcolor=fillcolor)

            if node.left:
                left_name = f"{name}L"
                render_node(node.left, left_name)
                dot.edge(name, left_name, label="True")
            if node.right:
                right_name = f"{name}R"
                render_node(node.right, right_name)
                dot.edge(name, right_name, label="False")

        render_node(self.root, "root")
        return dot
