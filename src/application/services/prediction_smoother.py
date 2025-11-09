import time
from collections import deque
from enum import Enum, auto

class AlertState(Enum):
    """Defines the 3-phase alert state."""
    NO_ALERT = auto()
    CONFIRMING = auto()
    ACTIVE = auto()

class PredictionSmoother:
    """
    Handles temporal smoothing of predictions and manages the 
    3-phase emergency alert system.
    """

    def __init__(
        self,
        window_size: int = 10,
        stability_threshold: int = 7,
        alert_confirm_time: float = 3.0,
        alert_active_time: float = 5.0
    ):
        """
        Initializes the service.

        Args:
            window_size: How many past predictions to store for smoothing.
            stability_threshold: How many times a gesture must appear in the
                                 window to be considered "stable".
            alert_confirm_time: Seconds to hold 'help_signal' to enter CONFIRMING.
            alert_active_time: Seconds to hold 'help_signal' to enter ACTIVE.
        """
        # For temporal smoothing
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.prediction_window: deque[str] = deque(maxlen=window_size)

        # For emergency alert
        self.alert_confirm_time = alert_confirm_time
        self.alert_active_time = alert_active_time
        self.alert_state = AlertState.NO_ALERT
        self.help_signal_start_time: float | None = None
        
        self.stable_prediction = "NONE"

    def _update_smoothing(self, raw_prediction: str) -> None:
        """Updates the smoothing window and finds a stable prediction."""
        self.prediction_window.append(raw_prediction)

        if len(self.prediction_window) < self.window_size:
            # Not enough data yet
            self.stable_prediction = "NONE"
            return

        # Find the most common prediction in the window
        try:
            counts = {pred: self.prediction_window.count(pred) for pred in set(self.prediction_window)}
            most_common_pred = max(counts, key=counts.get)

            # Check if it meets our stability threshold
            if counts[most_common_pred] >= self.stability_threshold:
                self.stable_prediction = most_common_pred
            else:
                self.stable_prediction = "NONE"
        except ValueError:
            self.stable_prediction = "NONE"

    def _update_alert_state(self, raw_prediction: str) -> None:
        """Updates the 3-phase emergency alert state machine."""
        current_time = time.time()

        if raw_prediction == 'help_signal':
            if self.help_signal_start_time is None:
                # First time seeing it
                self.help_signal_start_time = current_time
            
            # Check how long it's been held
            duration_held = current_time - self.help_signal_start_time
            
            if duration_held >= self.alert_active_time:
                self.alert_state = AlertState.ACTIVE
            elif duration_held >= self.alert_confirm_time:
                self.alert_state = AlertState.CONFIRMING
            else:
                # Still in the initial detection phase
                pass 
        else:
            # Not a help signal, reset everything
            self.help_signal_start_time = None
            self.alert_state = AlertState.NO_ALERT

    def update(self, raw_prediction: str) -> None:
        """
        Call this once per frame with the raw prediction from the model.
        """
        self._update_smoothing(raw_prediction)
        self._update_alert_state(raw_prediction)

    def get_stable_prediction(self) -> str:
        """
        Returns the stable, smoothed prediction (e.g., 'FIST' or 'NONE').
        """
        # We don't want the help signal to be "smoothed out"
        if self.alert_state != AlertState.NO_ALERT:
            return "HELP_SIGNAL"
            
        return self.stable_prediction

    def get_alert_state(self) -> AlertState:
        """
        Returns the current emergency alert state.
        """
        return self.alert_state