# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import numpy as np


class PolicyError(Exception):
    pass


class Policy:
    def __init__(self, policy_type: str, exploration: float):
        assert policy_type in ["alpha_zero", "other"]
        self.policy_type = policy_type
        self.exploration = exploration

    def _alpha_zero(
        self, q: np.ndarray, counts: np.ndarray, priors: np.ndarray
    ) -> np.ndarray:
        u = self.exploration * priors * (np.sqrt(counts.sum()) / (1 + counts))
        scores = q + u
        scores[scores == -math.inf] = 0
        total = scores.sum()  # we would have exited earlier if q was -inf everywhere
        return scores / total

    def _other(
        self, q: np.ndarray, counts: np.ndarray, priors: np.ndarray
    ) -> np.ndarray:
        """
        Monte-Carlo Tree Search as Regularized Policy Optimization
        https://arxiv.org/abs/2007.12509
        """
        tolerance = 1e-3
        multiplier = self.exploration * np.sqrt(counts.sum()) / (1 + counts).sum()
        mul_priors = multiplier * priors

        # Use binary search to find alpha such that abs(sum(pi_alpha) - 1) < tol
        # sum(pi_alpha) is strictly decreasing on alpha_min, alpha_max
        def find_alpha(alpha_min: float, alpha_max: float, depth: int = 0) -> float:
            if depth > 50:
                raise PolicyError("Binary search depth excedeed")
            if alpha_min > alpha_max:
                raise PolicyError(f"Binary search borked: {alpha_min} > {alpha_max}")
            alpha_mid = (alpha_min + alpha_max) / 2
            denom = alpha_mid - q
            if np.any(denom == 0):
                raise PolicyError(
                    f"Precision issue: alpha_mid={alpha_mid} q={q} denom={denom}"
                )
            pi_alpha_sum = (mul_priors / denom).sum()
            if abs(pi_alpha_sum - 1) < tolerance:
                return alpha_mid
            if pi_alpha_sum > 1:
                return find_alpha(alpha_mid, alpha_max, depth + 1)
            return find_alpha(alpha_min, alpha_mid, depth + 1)

        if multiplier > 0:
            alpha_min = (q + multiplier * priors).max().item()
            alpha_max = (q + multiplier).max().item()

            alpha = find_alpha(alpha_min, alpha_max)
            policy = mul_priors / (alpha - q)

            # This can happen if alpha_mid gets close to q
            if np.any(np.isnan(policy)):
                raise PolicyError("Got nan")
            return policy / policy.sum()
        else:
            q[q == -math.inf] = 0
            return q / q.sum()

    def get(self, q: np.ndarray, counts: np.ndarray, priors: np.ndarray) -> np.ndarray:
        assert priors.min() >= 0 and priors.max() <= 1 and (priors.sum() - 1) < 1e-6

        # if there is only one valid tactic, return it
        valid_ids = q != -math.inf
        n_valid = valid_ids.sum()
        assert n_valid >= 1, q
        if n_valid == 1:
            return valid_ids.astype(np.float64)

        # alpha zero
        if self.policy_type == "alpha_zero":
            return self._alpha_zero(q, counts, priors)

        # other policy (return alpha zero if fails)
        try:
            return self._other(q, counts, priors)
        except PolicyError as e:
            print(
                f"PolicyError: {e} -- q={q} - counts={counts} - priors={priors}",
                flush=True,
            )
            return self._alpha_zero(q, counts, priors)
