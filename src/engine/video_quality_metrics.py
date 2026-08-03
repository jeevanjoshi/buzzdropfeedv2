import math
import re
from typing import List, Dict, Any, Tuple


class VideoQualityMetrics:
    """
    Stage 8: Optical Flow & Fréchet Video Distance (FVD) Quality Verification Engine.
    Ensures rendered video clips pass two key spatiotemporal quality gates before upload:
    
    1. Fréchet Video Distance (FVD): Measures distributional distance between generated clip
       feature statistics and reference clip statistics. Lower is better.
       FVD = ||μ_g - μ_r||² + Tr(Σ_g + Σ_r - 2(Σ_g * Σ_r)^0.5)
    
    2. Optical Flow Temporal Consistency Score: Measures frame-to-frame motion smoothness
       to detect jump cuts, flickering, or incoherent temporal transitions.
    """

    FVD_THRESHOLD = 150.0          # FVD > 150 indicates poor perceptual video quality
    FLOW_CONSISTENCY_MIN = 0.70    # Flow consistency < 0.70 triggers editing alert

    def estimate_fvd_score(
        self,
        generated_feature_stats: Dict[str, float],
        reference_feature_stats: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Estimates Fréchet Video Distance (FVD) between generated and reference clip statistics.
        Uses simplified diagonal covariance approximation for lightweight CPU inference.

        Args:
            generated_feature_stats: {"mean": float, "variance": float} of generated clip I3D features
            reference_feature_stats: {"mean": float, "variance": float} of reference distribution
        """
        mu_g = generated_feature_stats.get("mean", 0.0)
        var_g = max(0.0, generated_feature_stats.get("variance", 1.0))
        mu_r = reference_feature_stats.get("mean", 0.0)
        var_r = max(0.0, reference_feature_stats.get("variance", 1.0))

        # Diagonal FVD approximation (scalar case):
        # FVD ≈ (μ_g - μ_r)² + (σ_g - σ_r)²  [simplified scalar version]
        mean_distance = (mu_g - mu_r) ** 2
        std_g = math.sqrt(var_g)
        std_r = math.sqrt(var_r)

        # Trace term: Tr(Σ_g + Σ_r - 2*sqrt(Σ_g * Σ_r))
        geometric_mean = math.sqrt(max(0.0, var_g * var_r))
        trace_term = var_g + var_r - 2.0 * geometric_mean

        fvd_score = mean_distance + trace_term
        passes = fvd_score <= self.FVD_THRESHOLD

        return {
            "fvd_score": round(fvd_score, 4),
            "mean_distance": round(mean_distance, 4),
            "trace_term": round(trace_term, 4),
            "threshold": self.FVD_THRESHOLD,
            "passes": passes,
            "verdict": "PASS — Visual quality within acceptable perceptual range." if passes
                       else f"FAIL — FVD {fvd_score:.1f} > {self.FVD_THRESHOLD}. Regenerate clip with refined prompt."
        }

    def estimate_optical_flow_consistency(self, frame_motion_vectors: List[float]) -> Dict[str, Any]:
        """
        Estimates per-clip temporal optical flow consistency score.
        Detects jump cuts, temporal aliasing, or flickering by measuring
        frame-to-frame motion vector variance normalized against clip mean motion.

        Args:
            frame_motion_vectors: List of per-frame motion magnitudes (e.g. from dense optical flow)
        """
        if len(frame_motion_vectors) < 2:
            return {
                "consistency_score": 1.0,
                "passes": True,
                "verdict": "Insufficient frames for flow analysis — assuming consistent."
            }

        mean_motion = sum(frame_motion_vectors) / len(frame_motion_vectors)
        if mean_motion < 1e-6:
            return {
                "consistency_score": 1.0,
                "passes": True,
                "verdict": "Static clip — zero motion detected, flow consistent."
            }

        # Coefficient of Variation (CV) as consistency proxy: lower CV = smoother motion
        variance = sum((m - mean_motion) ** 2 for m in frame_motion_vectors) / len(frame_motion_vectors)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_motion

        # Map CV to [0, 1] consistency score: 0 = highly erratic, 1 = perfectly smooth
        consistency_score = max(0.0, 1.0 - min(1.0, cv))
        passes = consistency_score >= self.FLOW_CONSISTENCY_MIN

        # Detect jump cut candidates: frames with > 2.5x mean motion magnitude
        jump_frames = [
            i for i, m in enumerate(frame_motion_vectors)
            if m > mean_motion * 2.5
        ]

        return {
            "consistency_score": round(consistency_score, 4),
            "coefficient_of_variation": round(cv, 4),
            "mean_motion_magnitude": round(mean_motion, 4),
            "jump_cut_frame_indices": jump_frames,
            "passes": passes,
            "threshold": self.FLOW_CONSISTENCY_MIN,
            "verdict": "PASS — Temporal motion is coherent and smooth." if passes
                       else f"FAIL — Flow consistency {consistency_score:.2f} < {self.FLOW_CONSISTENCY_MIN}. "
                            f"Jump cuts detected at frames: {jump_frames}. Re-render or blend transitions."
        }

    def run_full_quality_gate(
        self,
        shot_id: str,
        generated_feature_stats: Dict[str, float],
        reference_feature_stats: Dict[str, float],
        frame_motion_vectors: List[float]
    ) -> Dict[str, Any]:
        """
        Executes both Stage 8 quality gates (FVD + Optical Flow) for a single rendered shot.
        Returns a unified quality report for the shot.
        """
        fvd_result = self.estimate_fvd_score(generated_feature_stats, reference_feature_stats)
        flow_result = self.estimate_optical_flow_consistency(frame_motion_vectors)
        overall_pass = fvd_result["passes"] and flow_result["passes"]

        return {
            "shot_id": shot_id,
            "stage8_overall_pass": overall_pass,
            "fvd_gate": fvd_result,
            "optical_flow_gate": flow_result,
            "recommendation": "Ready for FFmpeg timeline assembly." if overall_pass
                              else "Re-render required: " + (
                                  fvd_result["verdict"] if not fvd_result["passes"] else flow_result["verdict"]
                              )
        }


video_quality_metrics = VideoQualityMetrics()
