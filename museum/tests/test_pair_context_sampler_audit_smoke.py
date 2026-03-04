from pathlib import Path

from execution.pair_context_sampler_audit_smoke import run_sampler_audit_smoke


def test_run_sampler_audit_smoke_writes_expected_artifacts(tmp_path: Path) -> None:
    summary = run_sampler_audit_smoke(
        runtime_dir=tmp_path / "sampler_smoke",
        exchange_target="coinbase_advanced",
        data_source="binance",
    )
    assert summary["schema"] == "moneyfan.pair_context_sampler_audit_smoke.v1"
    assert summary["results"]["conformance_result"] == "pass"
    assert summary["results"]["trace_rows_written"] >= 2
    assert summary["results"]["trace_report_rows_valid"] >= 2
    assert summary["results"]["focal_pair_inclusion_failures"] == 0

    for key in (
        "muxer_rows_jsonl",
        "trace_jsonl",
        "conformance_json",
        "trace_report_json",
        "trace_report_md",
        "summary_json",
    ):
        assert Path(summary["paths"][key]).exists()
