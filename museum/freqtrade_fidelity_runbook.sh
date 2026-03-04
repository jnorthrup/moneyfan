#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RUNTIME_DIR="${RUNTIME_DIR:-$ROOT_DIR/runtime}"
HOST="${HOST:-127.0.0.1}"
RECEIVER_PORT="${RECEIVER_PORT:-8091}"
CONTRACT_PROXY_PORT="${CONTRACT_PROXY_PORT:-8092}"
CONTRACT_PROXY_DOWNSTREAM_URL="${CONTRACT_PROXY_DOWNSTREAM_URL:-http://${HOST}:${RECEIVER_PORT}/trade-update}"
CONTRACT_PROXY_DOWNSTREAM_PAYLOAD_MODE="${CONTRACT_PROXY_DOWNSTREAM_PAYLOAD_MODE:-passthrough}"
BRIDGE_WEBHOOK_URL="${BRIDGE_WEBHOOK_URL:-http://${HOST}:${CONTRACT_PROXY_PORT}/signal}"
BRIDGE_RECEIVER_PROFILE="${BRIDGE_RECEIVER_PROFILE:-production_v1}"
EXCHANGE_TARGET="${EXCHANGE_TARGET:-coinbase_advanced}"
DATA_SOURCE="${DATA_SOURCE:-binance}"
BASELINE_RECON_JSON="${BASELINE_RECON_JSON:-}"
CANDIDATE_RECON_JSON="${CANDIDATE_RECON_JSON:-}"
DRY_RUN=0
PRINT_SUMMARY=1
TRAFFIC_REPLAY_BATCHES="${TRAFFIC_REPLAY_BATCHES:-3}"
TRAFFIC_REPLAY_BATCH_SIZE="${TRAFFIC_REPLAY_BATCH_SIZE:-4}"
TRAFFIC_VALIDATE_MIN_FORWARD_RATE="${TRAFFIC_VALIDATE_MIN_FORWARD_RATE:-1.0}"
TRAFFIC_VALIDATE_MAX_FILL_REJECTS="${TRAFFIC_VALIDATE_MAX_FILL_REJECTS:-0}"
TRAFFIC_VALIDATE_MAX_PROXY_REJECTS="${TRAFFIC_VALIDATE_MAX_PROXY_REJECTS:-0}"
TRAFFIC_VALIDATE_THRESHOLD_PROFILE="${TRAFFIC_VALIDATE_THRESHOLD_PROFILE:-${EXCHANGE_TARGET}__${DATA_SOURCE}}"
SAMPLER_VALIDATE_THRESHOLD_PROFILE="${SAMPLER_VALIDATE_THRESHOLD_PROFILE:-${EXCHANGE_TARGET}__${DATA_SOURCE}}"
VALIDATION_MODE="${VALIDATION_MODE:-0}"
SAMPLER_MUXER_ROWS_JSONL="${SAMPLER_MUXER_ROWS_JSONL:-$RUNTIME_DIR/pair_context_muxer_rows.jsonl}"
SAMPLER_TRACE_JSONL="${SAMPLER_TRACE_JSONL:-$RUNTIME_DIR/pair_context_sampler_trace.jsonl}"
SUBCOMMAND=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [global options] <subcommand>

Subcommands:
  receiver   Start local fill-event receiver (blocking)
  contract-proxy Start contract-compliant bridge receiver/proxy (blocking)
  bridge     Run handoff bridge (single pass by default)
  smoke      Run local contract path smoke test (bridge -> contract-proxy -> fill receiver)
  traffic-replay Run repeatable sample traffic replay through bridge -> contract-proxy -> fill receiver
  traffic-validate Run thresholded replay validation and emit JSON/markdown artifacts
  sampler-smoke Run end-to-end sampler audit smoke (sample muxer rows + traces + reports)
  sampler-validate Run thresholded sampler-audit validation and emit JSON/markdown artifacts
  sampler-profiles List available sampler validation threshold profiles
  sampler-conformance Validate muxer rows for pair-context sampler readiness
  sampler-trace-report Build JSON/markdown audit report from sampler trace JSONL
  sampler-audit Run sampler conformance, then trace report if conformance passes
  pipeline   Run bridge + normalize + reconcile one-shot pipeline
  replay     Re-run normalize + reconcile only (skip bridge)
  report     Render markdown report from reconciliation JSON
  compare    Render markdown compare report (baseline vs candidate reconciliation JSON)
  history    List timestamped report/compare snapshot history
  prune      Prune old runtime artifacts and timestamped snapshots
  all        Start receiver in background, run bridge + pipeline, then stop receiver
  print      Print resolved commands only

Global options:
  --runtime-dir PATH        Runtime artifact directory (default: $RUNTIME_DIR)
  --python BIN              Python executable (default: $PYTHON_BIN)
  --host HOST               Receiver bind host (default: $HOST)
  --receiver-port PORT      Receiver port (default: $RECEIVER_PORT)
  --contract-proxy-port P   Contract proxy port (default: $CONTRACT_PROXY_PORT)
  --contract-proxy-downstream-url URL Downstream webhook target for contract proxy (default: $CONTRACT_PROXY_DOWNSTREAM_URL)
  --contract-proxy-downstream-payload-mode M Downstream payload mapping mode (default: $CONTRACT_PROXY_DOWNSTREAM_PAYLOAD_MODE)
  --bridge-webhook-url URL  Bridge webhook target (default: $BRIDGE_WEBHOOK_URL)
  --bridge-receiver-profile P Bridge receiver profile (default: $BRIDGE_RECEIVER_PROFILE)
  --exchange-target X       Target execution venue label (default: $EXCHANGE_TARGET)
  --data-source X           Primary data source label (default: $DATA_SOURCE)
  --traffic-replay-batches N  Sample traffic replay batch count (default: $TRAFFIC_REPLAY_BATCHES)
  --traffic-replay-batch-size N Sample traffic replay batch size (default: $TRAFFIC_REPLAY_BATCH_SIZE)
  --traffic-validate-min-forward-rate X Threshold for replay validation (default: $TRAFFIC_VALIDATE_MIN_FORWARD_RATE)
  --traffic-validate-max-fill-rejects N Threshold for replay validation (default: $TRAFFIC_VALIDATE_MAX_FILL_REJECTS)
  --traffic-validate-max-proxy-rejects N Threshold for replay validation (default: $TRAFFIC_VALIDATE_MAX_PROXY_REJECTS)
  --traffic-validate-threshold-profile P Named threshold profile for replay validation (default: $TRAFFIC_VALIDATE_THRESHOLD_PROFILE)
  --sampler-validate-threshold-profile P Named threshold profile for sampler validation (default: $SAMPLER_VALIDATE_THRESHOLD_PROFILE)
  --validation-mode          In 'all', also run traffic-validate after report
  --no-validation-mode       In 'all', skip traffic-validate (default)
  --sampler-muxer-rows-jsonl P Input muxer rows JSONL for sampler conformance (default: $SAMPLER_MUXER_ROWS_JSONL)
  --sampler-trace-jsonl P    Input sampler trace JSONL for sampler report (default: $SAMPLER_TRACE_JSONL)
  --baseline-recon-json P   Baseline reconciliation JSON for compare subcommand
  --candidate-recon-json P  Candidate reconciliation JSON for compare subcommand
  --dry-run                 Print commands without executing
  --no-print-summary        Do not pass --print-summary to pipeline
  -h, --help                Show this help

Environment overrides:
  PYTHON_BIN, RUNTIME_DIR, HOST, RECEIVER_PORT, CONTRACT_PROXY_PORT, CONTRACT_PROXY_DOWNSTREAM_URL, CONTRACT_PROXY_DOWNSTREAM_PAYLOAD_MODE, BRIDGE_WEBHOOK_URL, BRIDGE_RECEIVER_PROFILE, EXCHANGE_TARGET, DATA_SOURCE, TRAFFIC_REPLAY_BATCHES, TRAFFIC_REPLAY_BATCH_SIZE, TRAFFIC_VALIDATE_MIN_FORWARD_RATE, TRAFFIC_VALIDATE_MAX_FILL_REJECTS, TRAFFIC_VALIDATE_MAX_PROXY_REJECTS, TRAFFIC_VALIDATE_THRESHOLD_PROFILE, SAMPLER_VALIDATE_THRESHOLD_PROFILE, VALIDATION_MODE, SAMPLER_MUXER_ROWS_JSONL, SAMPLER_TRACE_JSONL,
  BASELINE_RECON_JSON, CANDIDATE_RECON_JSON
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime-dir)
      RUNTIME_DIR="$2"; shift 2 ;;
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --host)
      HOST="$2"; shift 2 ;;
    --receiver-port)
      RECEIVER_PORT="$2"; shift 2 ;;
    --contract-proxy-port)
      CONTRACT_PROXY_PORT="$2"; shift 2 ;;
    --contract-proxy-downstream-url)
      CONTRACT_PROXY_DOWNSTREAM_URL="$2"; shift 2 ;;
    --contract-proxy-downstream-payload-mode)
      CONTRACT_PROXY_DOWNSTREAM_PAYLOAD_MODE="$2"; shift 2 ;;
    --bridge-webhook-url)
      BRIDGE_WEBHOOK_URL="$2"; shift 2 ;;
    --bridge-receiver-profile)
      BRIDGE_RECEIVER_PROFILE="$2"; shift 2 ;;
    --exchange-target)
      EXCHANGE_TARGET="$2"; shift 2 ;;
    --data-source)
      DATA_SOURCE="$2"; shift 2 ;;
    --traffic-replay-batches)
      TRAFFIC_REPLAY_BATCHES="$2"; shift 2 ;;
    --traffic-replay-batch-size)
      TRAFFIC_REPLAY_BATCH_SIZE="$2"; shift 2 ;;
    --traffic-validate-min-forward-rate)
      TRAFFIC_VALIDATE_MIN_FORWARD_RATE="$2"; shift 2 ;;
    --traffic-validate-max-fill-rejects)
      TRAFFIC_VALIDATE_MAX_FILL_REJECTS="$2"; shift 2 ;;
    --traffic-validate-max-proxy-rejects)
      TRAFFIC_VALIDATE_MAX_PROXY_REJECTS="$2"; shift 2 ;;
    --traffic-validate-threshold-profile)
      TRAFFIC_VALIDATE_THRESHOLD_PROFILE="$2"; shift 2 ;;
    --sampler-validate-threshold-profile)
      SAMPLER_VALIDATE_THRESHOLD_PROFILE="$2"; shift 2 ;;
    --validation-mode)
      VALIDATION_MODE=1; shift ;;
    --no-validation-mode)
      VALIDATION_MODE=0; shift ;;
    --sampler-muxer-rows-jsonl)
      SAMPLER_MUXER_ROWS_JSONL="$2"; shift 2 ;;
    --sampler-trace-jsonl)
      SAMPLER_TRACE_JSONL="$2"; shift 2 ;;
    --baseline-recon-json)
      BASELINE_RECON_JSON="$2"; shift 2 ;;
    --candidate-recon-json)
      CANDIDATE_RECON_JSON="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --no-print-summary)
      PRINT_SUMMARY=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    receiver|contract-proxy|bridge|smoke|traffic-replay|traffic-validate|sampler-smoke|sampler-validate|sampler-profiles|sampler-conformance|sampler-trace-report|sampler-audit|pipeline|replay|report|compare|history|prune|all|print)
      SUBCOMMAND="$1"; shift; break ;;
    *)
      echo "Unknown option or subcommand: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "${SUBCOMMAND}" ]]; then
  echo "Missing subcommand." >&2
  usage
  exit 2
fi

mkdir -p "$RUNTIME_DIR"

HANDOFF_PATH="$RUNTIME_DIR/freqtrade_handoff.jsonl"
BRIDGE_STATE_PATH="$RUNTIME_DIR/freqtrade_handoff_bridge_state.json"
ACK_LOG_PATH="$RUNTIME_DIR/freqtrade_dispatch_ack.jsonl"
RAW_FILL_UPDATES_PATH="$RUNTIME_DIR/freqtrade_trade_updates_raw.jsonl"
FILL_EVENTS_PATH="$RUNTIME_DIR/freqtrade_fill_events.jsonl"
FILL_REJECTS_PATH="$RUNTIME_DIR/freqtrade_fill_event_rejects.jsonl"
CONTRACT_PROXY_INGEST_PATH="$RUNTIME_DIR/freqtrade_contract_receiver_ingest.jsonl"
CONTRACT_PROXY_DISPATCH_PATH="$RUNTIME_DIR/freqtrade_contract_receiver_dispatch.jsonl"
CONTRACT_PROXY_REJECTS_PATH="$RUNTIME_DIR/freqtrade_contract_receiver_rejects.jsonl"
DISPATCH_LOG_PATH="$RUNTIME_DIR/hrm_fidelity_dispatch.jsonl"
RECON_JSON_PATH="$RUNTIME_DIR/hrm_freqtrade_fidelity_reconciliation.json"
RECON_CSV_PATH="$RUNTIME_DIR/hrm_freqtrade_fidelity_reconciliation.csv"
REPORT_MD_PATH="$RUNTIME_DIR/hrm_freqtrade_fidelity_report.md"
COMPARE_REPORT_MD_PATH="$RUNTIME_DIR/hrm_freqtrade_fidelity_compare_report.md"

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '+'
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    printf '\n'
  else
    "$@"
  fi
}

receiver_cmd=(
  "$PYTHON_BIN" -m execution.freqtrade_fill_event_receiver
  --host "$HOST"
  --port "$RECEIVER_PORT"
  --raw-log-path "$RAW_FILL_UPDATES_PATH"
  --fill-event-log-path "$FILL_EVENTS_PATH"
  --reject-log-path "$FILL_REJECTS_PATH"
)

contract_proxy_cmd=(
  "$PYTHON_BIN" -m execution.freqtrade_contract_receiver_proxy
  --host "$HOST"
  --port "$CONTRACT_PROXY_PORT"
  --ingest-log-path "$CONTRACT_PROXY_INGEST_PATH"
  --dispatch-log-path "$CONTRACT_PROXY_DISPATCH_PATH"
  --reject-log-path "$CONTRACT_PROXY_REJECTS_PATH"
  --downstream-webhook-url "$CONTRACT_PROXY_DOWNSTREAM_URL"
  --downstream-payload-mode "$CONTRACT_PROXY_DOWNSTREAM_PAYLOAD_MODE"
)

bridge_cmd=(
  "$PYTHON_BIN" -m execution.freqtrade_handoff_bridge
  --handoff-path "$HANDOFF_PATH"
  --state-path "$BRIDGE_STATE_PATH"
  --ack-log-path "$ACK_LOG_PATH"
  --webhook-url "$BRIDGE_WEBHOOK_URL"
  --receiver-profile "$BRIDGE_RECEIVER_PROFILE"
)

smoke_cmd=(
  "$PYTHON_BIN" -m execution.freqtrade_contract_path_smoke
  --runtime-dir "$RUNTIME_DIR/contract_path_smoke"
)

traffic_replay_cmd=(
  "$PYTHON_BIN" -m execution.freqtrade_contract_path_replay
  --runtime-dir "$RUNTIME_DIR/contract_path_replay"
  --batches "$TRAFFIC_REPLAY_BATCHES"
  --batch-size "$TRAFFIC_REPLAY_BATCH_SIZE"
  --exchange-target "$EXCHANGE_TARGET"
  --data-source "$DATA_SOURCE"
)

traffic_validate_cmd=(
  "$PYTHON_BIN" -m execution.freqtrade_contract_path_replay_validate
  --runtime-dir "$RUNTIME_DIR/contract_path_replay_validation"
  --batches "$TRAFFIC_REPLAY_BATCHES"
  --batch-size "$TRAFFIC_REPLAY_BATCH_SIZE"
  --exchange-target "$EXCHANGE_TARGET"
  --data-source "$DATA_SOURCE"
  --threshold-profile "$TRAFFIC_VALIDATE_THRESHOLD_PROFILE"
  --min-forward-rate "$TRAFFIC_VALIDATE_MIN_FORWARD_RATE"
  --max-fill-rejects "$TRAFFIC_VALIDATE_MAX_FILL_REJECTS"
  --max-proxy-rejects "$TRAFFIC_VALIDATE_MAX_PROXY_REJECTS"
  --replay-json-out "$RUNTIME_DIR/contract_path_replay_summary.json"
  --validation-json-out "$RUNTIME_DIR/contract_path_replay_validation.json"
  --validation-md-out "$RUNTIME_DIR/contract_path_replay_validation.md"
  --also-write-timestamped
)

pipeline_cmd=(
  "$PYTHON_BIN" -m execution.freqtrade_fidelity_pipeline
  --handoff-path "$HANDOFF_PATH"
  --bridge-state-path "$BRIDGE_STATE_PATH"
  --ack-log-path "$ACK_LOG_PATH"
  --raw-fill-updates-path "$RAW_FILL_UPDATES_PATH"
  --canonical-fill-events-path "$FILL_EVENTS_PATH"
  --normalizer-reject-log-path "$FILL_REJECTS_PATH"
  --dispatch-log-path "$DISPATCH_LOG_PATH"
  --reconciliation-json-path "$RECON_JSON_PATH"
  --reconciliation-csv-path "$RECON_CSV_PATH"
  --normalizer-reset-output
  --exchange-target "$EXCHANGE_TARGET"
  --data-source "$DATA_SOURCE"
)
if [[ "$PRINT_SUMMARY" -eq 1 ]]; then
  pipeline_cmd+=(--print-summary)
fi

replay_cmd=("${pipeline_cmd[@]}" --skip-bridge)

report_cmd=(
  "$PYTHON_BIN" -m execution.freqtrade_fidelity_report
  --reconciliation-json "$RECON_JSON_PATH"
  --out-md "$REPORT_MD_PATH"
  --also-write-timestamped
)

resolve_baseline_recon_json() {
  if [[ -n "$BASELINE_RECON_JSON" ]]; then
    printf '%s\n' "$BASELINE_RECON_JSON"
  else
    printf '%s\n' "$RECON_JSON_PATH"
  fi
}

resolve_candidate_recon_json() {
  if [[ -n "$CANDIDATE_RECON_JSON" ]]; then
    printf '%s\n' "$CANDIDATE_RECON_JSON"
  else
    printf '%s\n' "$RECON_JSON_PATH"
  fi
}

build_compare_cmd() {
  local baseline_json candidate_json
  baseline_json="$(resolve_baseline_recon_json)"
  candidate_json="$(resolve_candidate_recon_json)"
  compare_cmd=(
    "$PYTHON_BIN" -m execution.freqtrade_fidelity_compare_report
    --baseline-json "$baseline_json"
    --candidate-json "$candidate_json"
    --out-md "$COMPARE_REPORT_MD_PATH"
    --also-write-timestamped
  )
}

compare_cmd=()
build_compare_cmd

prune_cmd=(
  "$PYTHON_BIN" -m execution.freqtrade_fidelity_retention
  --runtime-dir "$RUNTIME_DIR"
  --dry-run
)

history_cmd=(
  "$PYTHON_BIN" -m execution.freqtrade_fidelity_history
  --runtime-dir "$RUNTIME_DIR"
)

sampler_conformance_cmd=(
  "$PYTHON_BIN" -m execution.pair_context_sampler_conformance
  --in-jsonl "$SAMPLER_MUXER_ROWS_JSONL"
  --out-json "$RUNTIME_DIR/pair_context_sampler_conformance.json"
  --exchange-target "$EXCHANGE_TARGET"
  --data-source "$DATA_SOURCE"
  --print-summary
)

sampler_smoke_cmd=(
  "$PYTHON_BIN" -m execution.pair_context_sampler_audit_smoke
  --runtime-dir "$RUNTIME_DIR/pair_context_sampler_audit_smoke"
  --exchange-target "$EXCHANGE_TARGET"
  --data-source "$DATA_SOURCE"
  --print-summary
)

sampler_validate_cmd=(
  "$PYTHON_BIN" -m execution.pair_context_sampler_audit_validate
  --runtime-dir "$RUNTIME_DIR/pair_context_sampler_audit_validation"
  --exchange-target "$EXCHANGE_TARGET"
  --data-source "$DATA_SOURCE"
  --threshold-profile "$SAMPLER_VALIDATE_THRESHOLD_PROFILE"
  --summary-json-out "$RUNTIME_DIR/pair_context_sampler_audit_smoke_summary.json"
  --validation-json-out "$RUNTIME_DIR/pair_context_sampler_audit_validation.json"
  --validation-md-out "$RUNTIME_DIR/pair_context_sampler_audit_validation.md"
  --also-write-timestamped
)

sampler_profiles_cmd=(
  "$PYTHON_BIN" -m execution.pair_context_sampler_audit_validate
  --list-threshold-profiles
)
if [[ "$PRINT_SUMMARY" -eq 0 ]]; then
  sampler_profiles_cmd+=(--print-json)
fi

sampler_trace_report_cmd=(
  "$PYTHON_BIN" -m execution.pair_context_sampler_trace_report
  --trace-jsonl "$SAMPLER_TRACE_JSONL"
  --out-json "$RUNTIME_DIR/pair_context_sampler_trace_report.json"
  --out-md "$RUNTIME_DIR/pair_context_sampler_trace_report.md"
  --also-write-timestamped
  --print-summary
)

print_commands() {
  echo "Resolved runtime directory: $RUNTIME_DIR"
  run_cmd "${receiver_cmd[@]}"
  run_cmd "${contract_proxy_cmd[@]}"
  run_cmd "${bridge_cmd[@]}"
  run_cmd "${smoke_cmd[@]}"
  run_cmd "${traffic_replay_cmd[@]}"
  run_cmd "${traffic_validate_cmd[@]}"
  run_cmd "${sampler_smoke_cmd[@]}"
  run_cmd "${sampler_validate_cmd[@]}"
  run_cmd "${sampler_profiles_cmd[@]}"
  run_cmd "${sampler_conformance_cmd[@]}"
  run_cmd "${sampler_trace_report_cmd[@]}"
  run_cmd "${pipeline_cmd[@]}"
  run_cmd "${replay_cmd[@]}"
  run_cmd "${report_cmd[@]}"
  run_cmd "${compare_cmd[@]}"
  run_cmd "${history_cmd[@]}"
  run_cmd "${prune_cmd[@]}"
}

case "$SUBCOMMAND" in
  print)
    DRY_RUN=1
    print_commands
    ;;
  receiver)
    run_cmd "${receiver_cmd[@]}"
    ;;
  contract-proxy)
    run_cmd "${contract_proxy_cmd[@]}"
    ;;
  bridge)
    run_cmd "${bridge_cmd[@]}"
    ;;
  smoke)
    run_cmd "${smoke_cmd[@]}"
    ;;
  traffic-replay)
    run_cmd "${traffic_replay_cmd[@]}"
    ;;
  traffic-validate)
    run_cmd "${traffic_validate_cmd[@]}"
    ;;
  sampler-smoke)
    run_cmd "${sampler_smoke_cmd[@]}"
    ;;
  sampler-validate)
    run_cmd "${sampler_validate_cmd[@]}"
    ;;
  sampler-profiles)
    run_cmd "${sampler_profiles_cmd[@]}"
    ;;
  sampler-conformance)
    run_cmd "${sampler_conformance_cmd[@]}"
    ;;
  sampler-trace-report)
    run_cmd "${sampler_trace_report_cmd[@]}"
    ;;
  sampler-audit)
    if [[ "$DRY_RUN" -eq 1 ]]; then
      run_cmd "${sampler_conformance_cmd[@]}"
      run_cmd "${sampler_trace_report_cmd[@]}"
    else
      echo "Running sampler conformance..."
      "${sampler_conformance_cmd[@]}"
      echo "Running sampler trace report..."
      "${sampler_trace_report_cmd[@]}"
    fi
    ;;
  pipeline)
    run_cmd "${pipeline_cmd[@]}"
    ;;
  replay)
    run_cmd "${replay_cmd[@]}"
    ;;
  report)
    run_cmd "${report_cmd[@]}"
    ;;
  compare)
    run_cmd "${compare_cmd[@]}"
    ;;
  history)
    run_cmd "${history_cmd[@]}"
    ;;
  prune)
    run_cmd "${prune_cmd[@]}"
    ;;
  all)
    if [[ "$DRY_RUN" -eq 1 ]]; then
      print_commands
      exit 0
    fi

    echo "Starting fill-event receiver in background..."
    "${receiver_cmd[@]}" &
    RECEIVER_PID=$!
    echo "Starting contract proxy in background..."
    "${contract_proxy_cmd[@]}" &
    CONTRACT_PROXY_PID=$!
    cleanup() {
      if [[ -n "${RECEIVER_PID:-}" ]] && kill -0 "$RECEIVER_PID" 2>/dev/null; then
        kill "$RECEIVER_PID" 2>/dev/null || true
        wait "$RECEIVER_PID" 2>/dev/null || true
      fi
      if [[ -n "${CONTRACT_PROXY_PID:-}" ]] && kill -0 "$CONTRACT_PROXY_PID" 2>/dev/null; then
        kill "$CONTRACT_PROXY_PID" 2>/dev/null || true
        wait "$CONTRACT_PROXY_PID" 2>/dev/null || true
      fi
    }
    trap cleanup EXIT INT TERM
    sleep 1

    echo "Running bridge..."
    "${bridge_cmd[@]}"
    echo "Running fidelity pipeline..."
    "${pipeline_cmd[@]}"
    echo "Rendering markdown report..."
    "${report_cmd[@]}"
    if [[ "$VALIDATION_MODE" -eq 1 ]]; then
      echo "Running thresholded traffic validation..."
      "${traffic_validate_cmd[@]}"
    fi
    echo "Pruning old artifacts (dry-run preview only)..."
    "${prune_cmd[@]}"
    ;;
esac
