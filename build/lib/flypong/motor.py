"""Descending-neuron spike counts to a paddle command in [-1, 1].

Negative moves the paddle up. Ball above the paddle drives the left eye,
which fires left descending neurons, which must move the paddle up, so
delta = right - left.
"""


REFERENCE_TOTAL = 3.0     # typical escape spikes per tick; normalized deltas match raw ones at this level
TOTAL_DECAY = 0.98        # slow running average of total escape activity (about 50 ticks)


class MotorReadout:
    def __init__(self, decay: float = 0.7, gain: float = 0.5, normalize: bool = False):
        self.decay = decay
        self.gain = gain
        self.normalize = normalize
        self.accumulator = 0.0
        self.total = REFERENCE_TOTAL

    def update(self, dn_left: int, dn_right: int) -> float:
        left, right = float(dn_left), float(dn_right)
        delta = right - left
        self.total = TOTAL_DECAY * self.total + (1.0 - TOTAL_DECAY) * (left + right)
        if self.normalize:
            # divide by the running total so global excitability changes do not change the command
            delta *= REFERENCE_TOTAL / max(self.total, 0.5)
        self.accumulator = self.decay * self.accumulator + delta
        return max(-1.0, min(1.0, self.accumulator * self.gain))

    def reset(self) -> None:
        self.accumulator = 0.0
        self.total = REFERENCE_TOTAL
