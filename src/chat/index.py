"""Future index boundary for canonical and delta content."""

from abc import ABC, abstractmethod

from src.canonical.model import DocumentCanonicalRepresentation


class CanonicalIndex(ABC):
    @abstractmethod
    def add(self, document: DocumentCanonicalRepresentation) -> None:
        """Index a canonical document without altering it."""
