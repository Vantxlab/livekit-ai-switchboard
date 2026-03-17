from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .context import Context


@dataclass
class Rule:
    """Developer-defined routing rule evaluated each turn."""

    name: str
    condition: Callable[[Context], bool]
    use: str
    priority: int = 5
