"""Descending-neuron spike counts to a paddle command in [-1, 1].

Negative moves the paddle up. Ball above the paddle drives the left eye,
which fires left descending neurons, which must move the paddle up, so
delta = right - left.
"""


class MotorReadout:
    def __init__(self, decay: float = 0.7, gain: float = 0.5):
        self.decay = decay
        self.gain = gain
        self.accumulator = 0.0

    def update(self, dn_left: int, dn_right: int) -> float:
        delta = float(dn_right) - float(dn_left)
        self.accumulator = self.decay * self.accumulator + delta
        return max(-1.0, min(1.0, self.accumulator * self.gain))

    def reset(self) -> None:
        self.accumulator = 0.0
