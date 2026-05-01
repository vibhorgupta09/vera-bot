#!/usr/bin/env bash
set -euo pipefail

rm -f /tmp/vera_llm_debug.log /tmp/vera_judge_last.log
python3 judge_simulator.py | tee /tmp/vera_judge_last.log
