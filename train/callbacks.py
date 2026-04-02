# train/callbacks.py

"""
Callback utilities for training.
Currently placeholder — early stopping and LR scheduling are
handled directly in the Trainer class.

Extend this module when adding:
    - Custom W&B artifact logging
    - Gradient histogram logging
    - Model architecture visualization
    - Hyperparameter sweep callbacks (Optuna integration)
"""


class EarlyStopping:
    """
    Simple early stopping tracker.
    Used as a standalone utility when not using the Trainer class.
    """
    def __init__(self, patience: int = 30, higher_is_better: bool = True):
        self.patience = patience
        self.higher_is_better = higher_is_better
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        is_better = (score > self.best_score if self.higher_is_better
                     else score < self.best_score)

        if is_better:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False
