"""Metric contracts for later canonical and delta evaluations."""

from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    @abstractmethod
    def compute(self, prediction: Any, reference: Any) -> float:
        """Return a normalized metric value."""
