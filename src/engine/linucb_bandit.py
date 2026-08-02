import numpy as np
from typing import List, Dict, Any, Tuple


class LinUCBContextualBandit:
    """
    Stage 5: LinUCB Contextual Multi-Armed Bandit Algorithm for Title & Packaging Optimization.
    Solves the Exploration-Exploitation Tradeoff using Ridge Regression and Upper Confidence Bounds:
    UCB_a = x^T * theta_a + alpha * sqrt(x^T * (A_a)^-1 * x)
    
    where A_a = F * F^T + lambda * I, and theta_a = (A_a)^-1 * b_a
    """

    def __init__(self, feature_dim: int = 5, alpha: float = 0.5, l2_reg: float = 1.0):
        self.d = feature_dim
        self.alpha = alpha
        self.l2_reg = l2_reg
        self.arms: Dict[str, Dict[str, Any]] = {}

    def register_arm(self, arm_id: str):
        """
        Registers a new title/thumbnail packaging arm in the bandit space.
        """
        if arm_id not in self.arms:
            self.arms[arm_id] = {
                "A": self.l2_reg * np.identity(self.d),
                "b": np.zeros((self.d, 1)),
                "trials": 0
            }

    def select_best_arm(self, context_vector: np.ndarray, candidate_arm_ids: List[str]) -> Tuple[str, float]:
        """
        Selects the optimal action (title/thumbnail variant) that maximizes the Upper Confidence Bound.
        """
        x = context_vector.reshape((self.d, 1))
        best_arm = None
        max_ucb = -float("inf")

        for arm_id in candidate_arm_ids:
            self.register_arm(arm_id)
            A_inv = np.linalg.inv(self.arms[arm_id]["A"])
            theta = np.dot(A_inv, self.arms[arm_id]["b"])

            # Expected reward estimate
            expected_reward = float(np.dot(x.T, theta)[0, 0])
            # Variance / Uncertainty bound
            variance_bound = float(np.sqrt(np.dot(np.dot(x.T, A_inv), x)[0, 0]))
            # LinUCB Score
            ucb_score = expected_reward + (self.alpha * variance_bound)

            if ucb_score > max_ucb:
                max_ucb = ucb_score
                best_arm = arm_id

        return best_arm or candidate_arm_ids[0], float(round(max_ucb, 4))

    def update_arm_reward(self, arm_id: str, context_vector: np.ndarray, reward: float):
        """
        Updates Ridge Regression matrices A and b with observed reward (e.g. CTR or Watch Time).
        """
        self.register_arm(arm_id)
        x = context_vector.reshape((self.d, 1))
        self.arms[arm_id]["A"] += np.dot(x, x.T)
        self.arms[arm_id]["b"] += reward * x
        self.arms[arm_id]["trials"] += 1


linucb_bandit = LinUCBContextualBandit()
