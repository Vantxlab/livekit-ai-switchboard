from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from .context import Context


@dataclass
class Rule:
    """Developer-defined routing rule evaluated each turn."""

    name: str
    condition: Callable[[Context], bool]
    use: Literal["fast", "smart"]
    priority: int = 5
    sticky: bool = False
