"""Future question-answering boundary."""

from abc import ABC, abstractmethod
from typing import Any


class AnswerService(ABC):
    @abstractmethod
    def answer(self, question: str) -> dict[str, Any]:
        """Answer against indexed canonical and delta information."""
