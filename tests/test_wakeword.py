"""
Tests for wakeword helpers.
"""

from src.server.wakeword import normalize_text, strip_wakeword


def test_normalize_text():
    assert normalize_text(" Hey,   Claw! ") == "hey claw"


def test_strip_wakeword_prefix():
    assert strip_wakeword("hey claw, what time is it", "hey claw") == "what time is it"


def test_strip_wakeword_first_match():
    assert strip_wakeword("Please hey claw open settings", "hey claw") == "Please open settings"
