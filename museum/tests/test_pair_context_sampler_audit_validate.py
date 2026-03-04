from execution.pair_context_sampler_audit_validate import (
    build_validation_markdown,
    resolve_threshold_profile,
    validate_sampler_audit_summary,
)


def _sample_summary() -> dict:
    return {
        "schema": "moneyfan.pair_context_sampler_audit_smoke.v1",
        "context": {"exchange_target": "coinbase_advanced", "data_source": "binance"},
        "results": {
            "conformance_result": "pass",
            "trace_rows_written": 2,
            "trace_report_rows_valid": 2,
            "pair_width_p95": 2.95,
            "focal_pair_inclusion_failures": 0,
        },
    }


def test_validate_sampler_audit_summary_pass() -> None:
    v = validate_sampler_audit_summary(
        _sample_summary(),
        min_trace_rows_valid=2,
        max_focal_pair_inclusion_failures=0,
        min_pair_width_p95=1.0,
        max_pair_width_p95=8.0,
        require_conformance_pass=True,
    )
    assert v["result"] == "pass"
    assert v["counts"]["trace_report_rows_valid"] == 2


def test_validate_sampler_audit_summary_fail_on_focal_failures() -> None:
    s = _sample_summary()
    s["results"]["focal_pair_inclusion_failures"] = 2
    v = validate_sampler_audit_summary(s, max_focal_pair_inclusion_failures=0)
    assert v["result"] == "fail"
    assert any(f["rule"] == "max_focal_pair_inclusion_failures" for f in v["failures"])


def test_profile_resolution_and_markdown_contains_context() -> None:
    p = resolve_threshold_profile(exchange_target="coinbase_advanced", data_source="binance")
    assert p["min_trace_rows_valid"] >= 1
    relaxed = resolve_threshold_profile(profile_name="coinbase_advanced__coinbase_advanced_relaxed")
    assert relaxed["max_pair_width_p95"] == 96.0
    v = validate_sampler_audit_summary(_sample_summary(), **p)
    v["profile"] = {"selected": "coinbase_advanced__binance", "resolved": p}
    md = build_validation_markdown(v, summary_json_path="/tmp/sampler_summary.json")
    assert "Sampler Audit Validation Report" in md
    assert "`exchange_target`" in md
    assert "coinbase_advanced__binance" in md
