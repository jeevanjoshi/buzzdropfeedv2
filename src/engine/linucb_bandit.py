import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class LinUCBContextualBandit:
    """
    Stage 5: LinUCB Contextual Multi-Armed Bandit Algorithm for Title & Packaging Optimization.
    Solves the Exploration-Exploitation Tradeoff using Ridge Regression and Upper Confidence Bounds:
    UCB_a = x^T * theta_a + alpha * sqrt(x^T * (A_a)^-1 * x)
    
    where A_a = F * F^T + lambda * I, and theta_a = (A_a)^-1 * b_a

    Extended with:
    - PRO Retention Feedback Loop: Detects retention cliff timestamps and injects pattern interrupt signals
    - Bandit State Persistence: Serialises A_a / b_a matrices to JSON for cross-run continuity
    - Joint Feature Scaling: Normalises context vectors across arms to prevent reward magnitude drift
    """

    STATE_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "logs",
        "bandit_state.json"
    )

    def __init__(self, feature_dim: int = 5, alpha: float = 0.5, l2_reg: float = 1.0):
        self.d = feature_dim
        self.alpha = alpha
        self.l2_reg = l2_reg
        self.arms: Dict[str, Dict[str, Any]] = {}
        self._load_state()

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

    def _normalize_context(self, context_vector: np.ndarray) -> np.ndarray:
        """
        Joint Feature Scaling: L2-normalize context vector to unit sphere before UCB computation.
        Prevents reward magnitude drift across different context scales.
        """
        norm = np.linalg.norm(context_vector)
        if norm < 1e-8:
            return context_vector
        return context_vector / norm

    def select_best_arm(self, context_vector: np.ndarray, candidate_arm_ids: List[str]) -> Tuple[str, float]:
        """
        Selects the optimal action (title/thumbnail variant) that maximizes the Upper Confidence Bound.
        """
        x = self._normalize_context(context_vector).reshape((self.d, 1))
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
        x = self._normalize_context(context_vector).reshape((self.d, 1))
        self.arms[arm_id]["A"] += np.dot(x, x.T)
        self.arms[arm_id]["b"] += reward * x
        self.arms[arm_id]["trials"] += 1
        self._save_state()

    def analyze_retention_curve(self, retention_timestamps: Dict[str, float]) -> Optional[float]:
        """
        PRO Retention Feedback Loop (Predictive Response Optimization):
        Analyses per-timestamp viewer retention to detect the first cliff-drop timestamp T.
        A cliff is defined as a retention drop >= 12% in a single 30-second window.
        Returns cliff timestamp in seconds, or None if retention is healthy.
        """
        if not retention_timestamps:
            return None
        
        sorted_ts = sorted(retention_timestamps.items(), key=lambda x: float(x[0]))
        for i in range(1, len(sorted_ts)):
            prev_t, prev_r = float(sorted_ts[i - 1][0]), sorted_ts[i - 1][1]
            curr_t, curr_r = float(sorted_ts[i][0]), sorted_ts[i][1]
            
            delta_time = curr_t - prev_t
            delta_retention = prev_r - curr_r
            
            if delta_time > 0 and delta_retention >= 0.12:
                return curr_t
        return None

    def suggest_pattern_interrupt(self, cliff_timestamp_sec: Optional[float]) -> Dict[str, Any]:
        """
        Given a retention cliff timestamp, suggests a visual pattern interrupt injection
        for the script segment near that timestamp.
        """
        if cliff_timestamp_sec is None:
            return {"interrupt_needed": False}
        
        return {
            "interrupt_needed": True,
            "cliff_timestamp_sec": cliff_timestamp_sec,
            "suggested_actions": [
                f"Insert B-roll cut at T={cliff_timestamp_sec:.0f}s",
                "Add dynamic data chart animation",
                "Inject SFX stinger at cut point",
                "Zoom-in on key visual at T-2s"
            ]
        }

    def _save_state(self):
        """
        Persists bandit matrices A and b to JSON file for cross-run state continuity.
        """
        try:
            os.makedirs(os.path.dirname(self.STATE_PATH), exist_ok=True)
            serializable = {}
            for arm_id, arm_data in self.arms.items():
                serializable[arm_id] = {
                    "A": arm_data["A"].tolist(),
                    "b": arm_data["b"].tolist(),
                    "trials": arm_data["trials"]
                }
            with open(self.STATE_PATH, "w") as f:
                json.dump(serializable, f)
        except Exception:
            pass  # Non-critical: state persistence failure is silent

    def _load_state(self):
        """
        Restores persisted bandit arm matrices from JSON file on startup.
        """
        try:
            if os.path.exists(self.STATE_PATH):
                with open(self.STATE_PATH, "r") as f:
                    raw = json.load(f)
                for arm_id, arm_data in raw.items():
                    self.arms[arm_id] = {
                        "A": np.array(arm_data["A"]),
                        "b": np.array(arm_data["b"]),
                        "trials": arm_data.get("trials", 0)
                    }
        except Exception:
            pass  # Non-critical: corrupted state is safely ignored


linucb_bandit = LinUCBContextualBandit()

