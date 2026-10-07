"""Tests for Module 8: SirenSmoother (no model needed - hand-made window decisions).

1 = siren window, 0 = other sound. One window per second (hop 1 s, window 2 s).
"""

import pytest

from app.audio.siren_classifier import SirenPrediction
from app.audio.siren_smoother import (
    NO_SIREN, SIREN_CONFIRMED, SIREN_SUSPECTED, SirenSmoother)


def window(is_siren, second, score=None, siren_type="Ambulance"):
    score = score if score is not None else (0.8 if is_siren else 0.01)
    return SirenPrediction(is_siren=bool(is_siren), siren_score=score,
                           siren_type=siren_type if is_siren else None,
                           type_scores={}, top_labels=[], inference_ms=1.0,
                           start_time_sec=float(second))


def run(pattern, **settings):
    smoother = SirenSmoother(**{"history_windows": 5, "windows_to_activate": 3,
                                "quiet_windows_to_end": 4, **settings})
    return [smoother.update(window(value, second)) for second, value in enumerate(pattern)]


def active_flags(states):
    return [int(state.emergency_active) for state in states]


def test_background_noise_never_activates():
    states = run([0] * 60)
    assert not any(active_flags(states))
    assert all(state.state == NO_SIREN for state in states)


def test_short_false_positive_is_ignored():
    # one or two isolated siren-like windows (e.g. a whistle) inside noise
    states = run([0] * 10 + [1] + [0] * 10 + [1, 1] + [0] * 10)
    assert not any(active_flags(states))
    assert states[10].state == SIREN_SUSPECTED          # noticed, but not confirmed
    assert states[-1].state == NO_SIREN


def test_continuous_siren_activates_after_three_windows_and_releases():
    states = run([0] * 5 + [1] * 20 + [0] * 10)
    flags = active_flags(states)
    assert flags.index(1) == 7                          # 3rd siren window (5, 6, 7)
    assert all(flags[7:25])                             # stays active for the whole siren
    assert flags[27] == 1 and flags[28] == 0            # 4th quiet window (25-28) ends it
    assert states[7].event_started_sec == pytest.approx(7 + 2.0)   # window 7 ends at 9 s
    assert states[-1].events_so_far == 1


def test_intermittent_siren_with_short_gap_is_one_event():
    # siren 10 s, 3 s gap (e.g. hidden behind a bus), siren 10 s
    states = run([0] * 3 + [1] * 10 + [0] * 3 + [1] * 10 + [0] * 8)
    flags = active_flags(states)
    first_on = flags.index(1)
    last_on = len(flags) - 1 - flags[::-1].index(1)
    assert all(flags[first_on:last_on + 1])             # no drop-out during the gap
    assert states[-1].events_so_far == 1


def test_gap_plus_one_model_miss_is_bridged_with_configured_release():
    # Measured case (Module 8 evaluation): a missed window just before a 3 s gap gave
    # 4 quiet windows in a row; with quiet_windows_to_end = 6 it stays ONE event
    pattern = [0] * 3 + [1] * 7 + [0] + [0] * 3 + [1] * 10 + [0] * 10
    assert run(pattern, quiet_windows_to_end=4)[-1].events_so_far == 2
    assert run(pattern, quiet_windows_to_end=6)[-1].events_so_far == 1


def test_config_values_bridge_gaps():
    from app.utils.config import load_config
    smoother = SirenSmoother.from_config(load_config())
    assert smoother.quiet_windows_to_end >= 6
    assert smoother.windows_to_activate <= smoother.history_windows


def test_slow_wail_alternating_windows_still_activates():
    # Module 7 finding: slow wail -> siren / other / siren / other ...
    states = run([0] * 3 + [1, 0] * 8 + [0] * 8)
    flags = active_flags(states)
    assert flags.index(1) == 7                          # 1,0,1,0,1 = 3 of last 5 at window 7
    assert all(flags[7:19])                             # max 1 quiet window in a row -> stays on
    assert states[-1].events_so_far == 1


def test_two_isolated_false_positives_close_together_are_ignored():
    states = run([0] * 5 + [1, 0, 0, 1] + [0] * 10)     # only 2 of any 5 windows
    assert not any(active_flags(states))


def test_long_gap_creates_two_separate_events():
    states = run([1] * 6 + [0] * 10 + [1] * 6 + [0] * 6)
    assert states[-1].events_so_far == 2


def test_smoothed_confidence_is_average_of_model_scores():
    smoother = SirenSmoother(history_windows=5, windows_to_activate=3, quiet_windows_to_end=4)
    scores = [0.92, 0.88, 0.91, 0.20]                   # the example from the project brief
    for second, score in enumerate(scores):
        state = smoother.update(window(score >= 0.5, second, score=score))
    assert state.smoothed_confidence == pytest.approx(sum(scores) / 4)
    assert state.state == SIREN_CONFIRMED               # 3 siren windows -> confirmed


def test_siren_type_is_most_frequent_during_event():
    smoother = SirenSmoother(history_windows=5, windows_to_activate=3, quiet_windows_to_end=4)
    types = ["Police", "Ambulance", "Ambulance", "Ambulance", "Police"]
    for second, siren_type in enumerate(types):
        state = smoother.update(window(1, second, siren_type=siren_type))
    assert state.siren_type == "Ambulance"


def test_after_release_old_sirens_do_not_retrigger():
    smoother = SirenSmoother(history_windows=5, windows_to_activate=3, quiet_windows_to_end=2)
    for second, value in enumerate([1, 1, 1, 0, 0]):     # confirmed, then released after 2 quiet
        state = smoother.update(window(value, second))
    assert not state.emergency_active
    state = smoother.update(window(0, 5))
    assert not state.emergency_active and state.history_length == 2


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        SirenSmoother(history_windows=3, windows_to_activate=4)
    with pytest.raises(ValueError):
        SirenSmoother(quiet_windows_to_end=0)
