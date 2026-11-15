"""An agent's decision settings, from its conversation subentry."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..const import (
    CONF_ANSWER_MODEL,
    CONF_DECIDER,
    CONF_DECIDER_MODEL,
    CONF_FALLBACK_AGENT,
    CONF_THRESHOLD,
    DEFAULT_DECIDER,
    DEFAULT_THRESHOLD,
)


@dataclass(frozen=True, slots=True)
class DeciderSettings:
    """An agent's decision settings."""

    enabled: bool  # the decision model goes first
    model: str | None  # its name on the decision server; None: the only one
    threshold: float
    answer_model: str | None  # words decided turns and splits several requests; None: the text model
    fallback_agent: str | None  # takes the turns the decider passes on; None: the agent loop

    @classmethod
    def from_subentry(cls, data: Mapping[str, Any]) -> DeciderSettings:
        """The settings as the subentry stores them, with the defaults for what it leaves out."""
        return cls(
            enabled=data.get(CONF_DECIDER, DEFAULT_DECIDER),
            model=data.get(CONF_DECIDER_MODEL) or None,
            threshold=float(data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)),
            answer_model=data.get(CONF_ANSWER_MODEL) or None,
            fallback_agent=data.get(CONF_FALLBACK_AGENT) or None,
        )
