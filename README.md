# CSVG — Autonomous YouTube Storytelling Video Pipeline

Autonomous 8-stage YouTube storytelling video generation pipeline (CSVG): RSS ingestion →
topic TOPSIS selection → RAG-grounded 6-act documentary script → LLM editing/auditing →
TTS + AI visuals + ffmpeg assembly → YouTube publish, targeting $2,000+/month ad revenue.

## Where things live (single source of truth)

- **Design, architecture, revenue model, and feature status (shipped / planned / not feasible):**
  `docs/PIPELINE_CONVERGED_PLAN.md`
- **Operational quick-reference** (commands, flags, tests, env & secrets, conventions):
  `AGENTS.md`
- **Environment template:** `example.env`
- **Tests:** `python run_tests.py` (hermetic end-to-end, no network/Pi/`.env` required)
- **Production run:** `./run_production.sh` (pre-flight health check, then background run + email)

Plain Python on the repo root; no build step. GitHub issues/PRs welcome for the planned items in
`docs/PIPELINE_CONVERGED_PLAN.md`.