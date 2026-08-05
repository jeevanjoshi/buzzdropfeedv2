import os
import json
import datetime


class MediaBudget:
    """
    Per-month paid image-generation budget guard (fal.ai / Replicate).

    Defaults to ~USD 24 / month (approx. INR 2000). Persists spend to a JSON file
    so it survives across pipeline runs within the same calendar month. Once the
    cap is reached (or nearly), `economy_mode()` returns True and the media
    producer switches entirely to FREE visual assets (Pixabay / matplotlib / SVG /
    synthetic), protecting the AI spend ceiling.

    All-or-nothing switching only — no per-shot adaptive logic.
    """

    MONTHLY_CAP_USD = 24.0  # ~ INR 2000 / month
    PAID_IMAGE_COST_USD = 0.003  # worst-case fal flux/schnell per image
    _RESERVE_USD = 1.0  # stop charging when remaining drops below this

    def __init__(self, file_path: str = "media_budget.json"):
        self._file = file_path
        self._data = self._load()

    def _load(self) -> dict:
        try:
            if os.path.exists(self._file):
                with open(self._file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError, OSError):
            pass
        return {}

    def _save(self) -> None:
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except (IOError, OSError):
            pass

    @staticmethod
    def _month_key() -> str:
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")

    def month_spend(self) -> float:
        return float(self._data.get(self._month_key(), {}).get("spend_usd", 0.0))

    def remaining(self) -> float:
        return max(0.0, self.MONTHLY_CAP_USD - self.month_spend())

    def economy_mode(self) -> bool:
        return self.month_spend() >= self.MONTHLY_CAP_USD or self.remaining() <= self._RESERVE_USD

    def charge_paid_image(self, cost_usd: float = PAID_IMAGE_COST_USD) -> bool:
        """Charges one paid image if within budget. Returns True if charged (use paid), else False."""
        if self.economy_mode():
            return False
        self._data[self._month_key()] = {
            "spend_usd": round(self.month_spend() + cost_usd, 4),
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self._save()
        return True

    def reset_month(self):
        self._data[self._month_key()] = {"spend_usd": 0.0}
        self._save()


media_budget = MediaBudget()
