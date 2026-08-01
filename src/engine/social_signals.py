import numpy as np
from typing import List, Dict, Any


def calculate_social_hype_multiplier(
    post_count_last_6h: int, avg_post_count_24h: float
) -> float:
    """
    Calculates Social Media Hype Multiplier (SHM) from X/Twitter and Reddit activity.
    SHM = (posts_in_last_6h * 4) / avg_24h_posts
    SHM > 1.5 indicates an exploding social conversation.
    """
    if avg_post_count_24h < 1e-3:
        return 1.0

    extrapolated_rate = float(post_count_last_6h * 4.0)
    hype_ratio = extrapolated_rate / float(avg_post_count_24h)
    
    # Bound SHM between 0.5 and 3.0
    shm_score = max(0.5, min(3.0, float(np.round(hype_ratio, 4))))
    return shm_score


def calculate_youtube_vph_velocity(
    recent_video_views: List[int], video_ages_hours: List[float]
) -> float:
    """
    Calculates YouTube Competitor Views-Per-Hour (VPH) Velocity.
    VPH = Total Views across recent competitor videos / Total Hours published
    VPH > 500 views/hr signals an active YouTube recommendation wave.
    """
    if not recent_video_views or not video_ages_hours:
        return 1.0  # Default baseline multiplier

    total_views = sum(recent_video_views)
    total_hours = max(1.0, sum(video_ages_hours))
    
    avg_vph = total_views / total_hours
    
    # Scale VPH score into a normalized TOPSIS factor [0.5, 3.0]
    vph_factor = 0.5 + (avg_vph / 250.0)
    return float(np.round(min(3.0, vph_factor), 4))
