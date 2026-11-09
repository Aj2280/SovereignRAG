"""Unit tests for hardware auto-detection."""

from sovereignrag.core.hardware import detect_apple_silicon, detect_ram, get_hardware_profile


def test_hardware_profile():
    profile = get_hardware_profile()
    assert profile.total_ram_gb > 0
    assert profile.os_name in ("Darwin", "Linux", "Windows")
    assert profile.recommended_device in ("mps", "cuda", "cpu")
    assert profile.recommended_profile in ("balanced", "deep_reasoning", "speed_demon")

    summary = profile.summary()
    assert "total_ram_gb" in summary
    assert "recommended_device" in summary
