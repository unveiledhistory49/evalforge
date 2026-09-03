#!/usr/bin/env bash
# EvalForge end-to-end smoke test: synthesize -> register -> run two models ->
# reproduce -> compare -> gate, all through the real CLI in a temp store.
set -euo pipefail

TMP="$(mktemp -d)"
export EVALFORGE_STORE="$TMP/store" PYTHONHASHSEED=0
trap 'rm -rf "$TMP"' EXIT
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/examples:$ROOT/src:${PYTHONPATH:-}"

echo "==> init + synthesize + register"
python3 -m evalforge.cli init >/dev/null
python3 -m evalforge.cli synthesize --path "$TMP/demo.jsonl" --n 120 --seed 4 >/dev/null
REG="$(python3 -m evalforge.cli register-dataset demo "$TMP/demo.jsonl")"
echo "$REG" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["version"] == 1 and d["n"] == 120, d'

echo "==> run threshold model"
RUN_T="$(python3 -m evalforge.cli run --model models:threshold_model --dataset demo --seed 7 --metrics accuracy,f1_binary --name t)"
ACC="$(echo "$RUN_T" | python3 -c 'import json,sys; print(json.load(sys.stdin)["aggregates"]["accuracy"])')"
python3 -c "import sys; assert float('$ACC') > 0.8, '$ACC'"
TID="$(echo "$RUN_T" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"

echo "==> reproduce (must match)"
python3 -m evalforge.cli reproduce "$TID" --model models:threshold_model | \
  python3 -c 'import json,sys; assert json.load(sys.stdin)["match"] is True'

echo "==> run constant baseline + compare"
RUN_C="$(python3 -m evalforge.cli run --model models:constant_zero --dataset demo --seed 7 --metrics accuracy --name c)"
CID="$(echo "$RUN_C" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
python3 -m evalforge.cli compare "$TID" "$CID" --metric accuracy --seed 2 --resamples 300 | \
  python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["diff_b_minus_a"] < 0 and d["wins"] > d["losses"], d'

echo "==> gate pass + fail"
python3 -m evalforge.cli gate --run "$TID" --rules '{"accuracy": {"min": 0.8}}' | \
  python3 -c 'import json,sys; assert json.load(sys.stdin)["passed"] is True'
if python3 -m evalforge.cli gate --run "$TID" --rules '{"accuracy": {"min": 0.99}}' >/dev/null 2>&1; then
  echo "FAIL: gate should have failed"; exit 1
fi
echo "E2E PASS"
