#!/usr/bin/env python3
"""
CSVG test runner — runs the single hermetic end-to-end suite.

There is exactly ONE test module: tests/test_hermetic_e2e.py — a fully
self-sufficient, no-network end-to-end test of the REAL orchestrator with the
REAL StoryDesigner + Observer agents (FakeLLM + boundary stubs only). If a case
fails there is no flaky mock to blame; it is a genuine pipeline regression.

Run:
    python run_tests.py
    python tests/test_hermetic_e2e.py
"""
import sys

from tests.test_hermetic_e2e import main

if __name__ == "__main__":
    sys.exit(main())