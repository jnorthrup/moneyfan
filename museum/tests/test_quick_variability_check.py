import os
import tempfile
import numpy as np
import pytest

from quick_variability_check import extract_losses, analyze_variability, get_recommendation, main


def test_extract_losses():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("INFO: pred_loss=123.45\n")
        f.write("DEBUG: something else\n")
        f.write("INFO: pred_loss=67.89\n")
        temp_path = f.name

    try:
        losses = extract_losses(temp_path)
        assert losses == [123.45, 67.89]
    finally:
        os.unlink(temp_path)


def test_extract_losses_empty_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("INFO: no losses here\n")
        temp_path = f.name

    try:
        losses = extract_losses(temp_path)
        assert losses == []
    finally:
        os.unlink(temp_path)


def test_analyze_variability():
    losses = [100.0, 150.0, 200.0]
    result = analyze_variability(losses)

    assert result['n'] == 3
    assert result['mean'] == 150.0
    assert result['min'] == 100.0
    assert result['max'] == 200.0
    assert result['median'] == 150.0
    # std deviation of [100, 150, 200] is roughly 40.82
    np.testing.assert_approx_equal(result['std'], 40.8248, significant=4)
    # CV = std / mean
    np.testing.assert_approx_equal(result['cv'], 40.8248 / 150.0, significant=4)


def test_analyze_variability_empty():
    assert analyze_variability([]) == {}


def test_get_recommendation():
    # Test Severely underpowered
    assert get_recommendation({'mean': 450, 'max': 850, 'cv': 0.1}) == "🔥 SEVERELY UNDERPOWERED - Increase hidden_dim to 128-256"

    # Test Underpowered
    assert get_recommendation({'mean': 250, 'max': 550, 'cv': 0.1}) == "⚠️ UNDERPOWERED - Consider testing hidden_dim=128"

    # Test Optimal
    assert get_recommendation({'mean': 40, 'max': 90, 'cv': 0.1}) == "✅ OPTIMAL/SLIGHTLY OVERPOWERED"

    # Test High Variability
    assert get_recommendation({'mean': 150, 'max': 200, 'cv': 0.6}) == "📈 HIGH VARIABILITY - Add more layers (3-4 each)"

    # Test Low Variability
    assert get_recommendation({'mean': 150, 'max': 200, 'cv': 0.1}) == "📉 LOW VARIABILITY - Model converging well"

    # Test Moderate
    assert get_recommendation({'mean': 150, 'max': 200, 'cv': 0.3}) == "✅ MODERATE - Current config may be adequate"

    # Test Empty
    assert get_recommendation({}) == "No data"


def test_main_refuses_to_fabricate_analysis(capsys):
    rc = main(["--log-file", "/definitely/missing.log"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "refusing to fabricate sizing guidance" in captured.out
