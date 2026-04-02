#!/bin/bash
# docker/entrypoint.sh
# Gate: env validation → smoke → unit tests → full suite. Abort on any failure.

set -euo pipefail   # -e = exit on error, -u = error on unset var, -o pipefail

WORKSPACE="/workspace"
LOG_DIR="${WORKSPACE}/logs"
RESULTS_DIR="${WORKSPACE}/results"
CKPT_DIR="${WORKSPACE}/checkpoints"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CKPT_DIR"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
ok()   { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓  $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠  $1${NC}"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] ✗  $1${NC}"; exit 1; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║          EdgeMamba-3 Experiment Suite                    ║${NC}"
echo -e "${BOLD}║          Yajat Kapur · 22BBS0110 · CBS1904               ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
log "Container started: $(date)"
log "CUDA devices: ${CUDA_VISIBLE_DEVICES:-all}"

# Quick GPU check
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" || true

# ════════════════════════════════════════════════════════════════════════════
# GATE 0: Environment validation
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}── Gate 0: Environment Validation ──────────────────────────────${NC}"
log "Running validate_env.py..."

python "$WORKSPACE/scripts/validate_env.py" 2>&1 | tee "$LOG_DIR/validate_env.log"
VALIDATE_EXIT=${PIPESTATUS[0]}

if [ $VALIDATE_EXIT -ne 0 ]; then
    fail "Environment validation failed. Check $LOG_DIR/validate_env.log"
fi
ok "Environment validated."

# ════════════════════════════════════════════════════════════════════════════
# GATE 1: Smoke Tests
# Run tiny 3-epoch models to verify the pipeline works end-to-end
# before committing to multi-hour training runs.
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}── Gate 1: Smoke Tests (3-epoch quick runs) ─────────────────────${NC}"
log "Smoke testing LRGB pipeline (Peptides-func, 3 epochs, d_model=32)..."

python -u "$WORKSPACE/scripts/smoke_test_lrgb.py" 2>&1 | tee "$LOG_DIR/smoke_lrgb.log"
SMOKE1_EXIT=${PIPESTATUS[0]}

if [ $SMOKE1_EXIT -ne 0 ]; then
    fail "LRGB smoke test failed. Check $LOG_DIR/smoke_lrgb.log"
fi
ok "LRGB smoke test passed."

log "Smoke testing RelBench pipeline (rel-hm user-churn, 3 epochs, d_model=32)..."

python -u "$WORKSPACE/scripts/smoke_test_relbench.py" 2>&1 | tee "$LOG_DIR/smoke_relbench.log"
SMOKE2_EXIT=${PIPESTATUS[0]}

if [ $SMOKE2_EXIT -ne 0 ]; then
    fail "RelBench smoke test failed. Check $LOG_DIR/smoke_relbench.log"
fi
ok "RelBench smoke test passed."

# ════════════════════════════════════════════════════════════════════════════
# GATE 2: Unit Tests
# Full pytest suite. Smoke passing guarantees the pipeline runs;
# unit tests guarantee every module's contract is met.
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}── Gate 2: Unit Tests ───────────────────────────────────────────${NC}"
log "Running pytest suite..."

python -m pytest "$WORKSPACE/tests/" -v --tb=short \
    --junitxml="$LOG_DIR/pytest_results.xml" \
    2>&1 | tee "$LOG_DIR/pytest.log"
TEST_EXIT=${PIPESTATUS[0]}

if [ $TEST_EXIT -ne 0 ]; then
    fail "Unit tests failed. Check $LOG_DIR/pytest.log and $LOG_DIR/pytest_results.xml"
fi
ok "All unit tests passed."

# ════════════════════════════════════════════════════════════════════════════
# GATE 3: Main Experiments
# Only reached if both smoke tests and unit tests pass.
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}── Gate 3: Main Experiments ─────────────────────────────────────${NC}"
log "All gates passed. Starting main experiments..."

run_experiment() {
    local script="$1"
    local config="$2"
    local log_file="$3"
    log "Running: $script --config $config"
    python -u "$WORKSPACE/$script" --config "$WORKSPACE/$config" \
        2>&1 | tee "$LOG_DIR/$log_file"
    local EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        fail "Experiment failed: $script $config — check $LOG_DIR/$log_file"
    fi
    ok "Completed: $config"
}

run_experiment "scripts/train_lrgb.py"      "configs/lrgb_peptides_func.yaml"   "train_pf.log"
run_experiment "scripts/train_lrgb.py"      "configs/lrgb_peptides_struct.yaml" "train_ps.log"
run_experiment "scripts/train_relbench.py"  "configs/relbench_hm_churn.yaml"    "train_hm.log"
run_experiment "scripts/train_relbench.py"  "configs/relbench_amazon_ltv.yaml"  "train_amz.log"

# ════════════════════════════════════════════════════════════════════════════
# GATE 4: Ablation Studies
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}── Gate 4: Ablation Studies ─────────────────────────────────────${NC}"
log "Running all 6 ablation studies (this will take several hours)..."

python -u "$WORKSPACE/ablations/run_ablations.py" \
    2>&1 | tee "$LOG_DIR/ablations.log"
ABL_EXIT=${PIPESTATUS[0]}

if [ $ABL_EXIT -ne 0 ]; then
    fail "Ablation run failed. Check $LOG_DIR/ablations.log"
fi

# Copy results to results dir
cp "$WORKSPACE/ablation_results.csv" "$RESULTS_DIR/ablation_results.csv" 2>/dev/null || true

# ── Final Summary ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║              All Experiments Complete                    ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
log "Finished: $(date)"
log "Logs:        $LOG_DIR/"
log "Checkpoints: $CKPT_DIR/"
log "Results:     $RESULTS_DIR/ablation_results.csv"
echo ""

# Print the ablation results table
if [ -f "$RESULTS_DIR/ablation_results.csv" ]; then
    echo -e "${BOLD}Ablation Results Summary:${NC}"
    python -c "
import csv
with open('/workspace/results/ablation_results.csv') as f:
    rows = list(csv.DictReader(f))
print(f\"{'Ablation':<25} {'Description':<45} {'Mean':>7} {'Std':>6}\")
print('-' * 90)
for r in rows:
    print(f\"{r['ablation']:<25} {r['description'][:44]:<45} {r['mean']:>7} {r['std']:>6}\")
"
fi
