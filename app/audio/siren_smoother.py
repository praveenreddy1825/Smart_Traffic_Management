"""
Module 8 - Audio temporal smoothing.

A single audio window must NOT trigger an emergency. This module turns the
per-window decisions of the siren classifier (Module 7) into a stable event:

    smoother = SirenSmoother.from_config(config)
    state = smoother.update(prediction)       # once per audio window
    state.emergency_active                    # True only when the siren is CONFIRMED

States and rules (values from config.yaml -> smoothing):

    NO_SIREN        --(any siren window in history)--------->  SIREN_SUSPECTED
    SIREN_SUSPECTED --(>= windows_to_activate of history)--->  SIREN_CONFIRMED  (emergency ACTIVE)
    SIREN_SUSPECTED --(history has no siren windows)-------->  NO_SIREN         (false alarm ignored)
    SIREN_CONFIRMED --(quiet_windows_to_end quiet in a row)->  NO_SIREN         (emergency ENDS)

Why "k of the last n" and not "n in a row": a slow wail siren can be heard as
siren / not-siren in alternating 2 s windows (measured in Module 7).
Why a separate, longer release rule (hysteresis): a siren briefly hidden by
traffic noise must not switch the emergency off and on again.
"""

from collections import Counter, deque
from dataclasses import dataclass

NO_SIREN = "NO_SIREN"
SIREN_SUSPECTED = "SIREN_SUSPECTED"
SIREN_CONFIRMED = "SIREN_CONFIRMED"


@dataclass
class AudioEmergencyState:
    state: str                    # NO_SIREN / SIREN_SUSPECTED / SIREN_CONFIRMED
    emergency_active: bool        # True only in SIREN_CONFIRMED
    window_is_siren: bool         # raw decision of the latest window (Module 7)
    window_score: float           # raw siren score of the latest window
    siren_windows_in_history: int
    history_length: int
    smoothed_confidence: float    # average model siren score over the history
    siren_type: str | None        # most frequent type during the current event
    event_started_sec: float | None   # stream time when the current event was confirmed
    event_duration_sec: float     # how long the confirmed event has lasted
    timestamp_sec: float          # end time of the latest window
    events_so_far: int            # number of confirmed events since start

    def text(self):
        status = "ACTIVE" if self.emergency_active else "INACTIVE"
        return (f"Emergency (audio): {status} [{self.state}] | siren windows "
                f"{self.siren_windows_in_history}/{self.history_length} | "
                f"smoothed confidence {self.smoothed_confidence:.2f}")


class SirenSmoother:
    def __init__(self, history_windows=5, windows_to_activate=3, quiet_windows_to_end=6,
                 window_duration_sec=2.0):
        if not 1 <= windows_to_activate <= history_windows:
            raise ValueError("smoothing: need 1 <= windows_to_activate <= history_windows")
        if quiet_windows_to_end < 1:
            raise ValueError("smoothing: quiet_windows_to_end must be >= 1")
        self.history_windows = history_windows
        self.windows_to_activate = windows_to_activate
        self.quiet_windows_to_end = quiet_windows_to_end
        self.window_duration_sec = window_duration_sec
        self.reset()

    @classmethod
    def from_config(cls, config):
        settings = config["smoothing"]
        return cls(
            history_windows=settings.get("history_windows", 5),
            windows_to_activate=settings.get("windows_to_activate", 3),
            quiet_windows_to_end=settings.get("quiet_windows_to_end", 6),
            window_duration_sec=config.get("audio", {}).get("window_seconds", 2.0),
        )

    def reset(self):
        self.state = NO_SIREN
        self._history = deque(maxlen=self.history_windows)   # (is_siren, score)
        self._quiet_streak = 0
        self._event_started_sec = None
        self._event_types = Counter()
        self.events_so_far = 0

    def update(self, prediction):
        """Feed one SirenPrediction (Module 7); returns the smoothed AudioEmergencyState."""
        self._history.append((prediction.is_siren, prediction.siren_score))
        self._quiet_streak = 0 if prediction.is_siren else self._quiet_streak + 1
        siren_count = sum(is_siren for is_siren, _ in self._history)
        # A window is "known" once it has ended: start + window length
        now = prediction.start_time_sec + self.window_duration_sec

        if self.state == SIREN_CONFIRMED:
            if self._quiet_streak >= self.quiet_windows_to_end:
                self._end_event(prediction)
                siren_count = 0
        elif siren_count >= self.windows_to_activate:
            self.state = SIREN_CONFIRMED
            self._event_started_sec = now
            self._event_types.clear()
            self.events_so_far += 1
        elif siren_count > 0:
            self.state = SIREN_SUSPECTED
        else:
            self.state = NO_SIREN

        active = self.state == SIREN_CONFIRMED
        if active and prediction.is_siren and prediction.siren_type:
            self._event_types[prediction.siren_type] += 1

        return AudioEmergencyState(
            state=self.state,
            emergency_active=active,
            window_is_siren=prediction.is_siren,
            window_score=prediction.siren_score,
            siren_windows_in_history=siren_count,
            history_length=len(self._history),
            smoothed_confidence=sum(score for _, score in self._history) / len(self._history),
            siren_type=self._event_types.most_common(1)[0][0] if (active and self._event_types)
            else None,
            event_started_sec=self._event_started_sec if active else None,
            event_duration_sec=(now - self._event_started_sec) if active else 0.0,
            timestamp_sec=now,
            events_so_far=self.events_so_far,
        )

    def _end_event(self, latest_prediction):
        self.state = NO_SIREN
        self._event_started_sec = None
        self._event_types.clear()
        # Start fresh (only the latest, quiet window) so old siren windows still in the
        # history cannot instantly re-trigger the emergency
        self._history.clear()
        self._history.append((latest_prediction.is_siren, latest_prediction.siren_score))
