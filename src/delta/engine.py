"""Delta-engine interface; comparison logic is intentionally deferred."""

from abc import ABC, abstractmethod

from src.canonical.model import DocumentCanonicalRepresentation


class DeltaEngine(ABC):
    @abstractmethod
    def compare(
        self,
        baseline: DocumentCanonicalRepresentation,
        revision: DocumentCanonicalRepresentation,
    ) -> dict:
        """Compare text, raw geometry, and residual visual layers."""
