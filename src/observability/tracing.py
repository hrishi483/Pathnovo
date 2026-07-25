"""Provider-neutral tracing boundary."""

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def span(name: str) -> Iterator[None]:
    """No-op span that can later be replaced by OpenTelemetry."""
    del name
    yield
