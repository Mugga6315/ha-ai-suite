r"""The loop's system prompt in the parts Home Assistant joins it from.

HA joins `user_prompt \n api_prompt [\n time_line] [\n extra_prompt]` (conversation/chat_log.py); the time line is
left out when a GetDateTime tool exists. No Home Assistant imports: the trace monitor's record uses it too.
"""

import re
from typing import NamedTuple

# HA's date and time line as rendered (helpers/llm.py DATE_TIME_PROMPT), after the "\n" HA joins the prompt's parts
# with; found at the prompt's end when no API prompt locates it.
_TIME_LINE = re.compile(r"\nCurrent time is \d{2}:\d{2}:\d{2}\. Today's date is \d{4}-\d{2}-\d{2}\.\Z")


class Parts(NamedTuple):
    """A system prompt's parts in their order; None: the prompt has no such part."""

    stable: str  # the user's prompt and the API's part: the prompt starts with it
    time_line: str | None
    extra: str | None


def split(prompt: str, api_prompt: str | None, extra_prompt: str | None) -> Parts:
    """The prompt's parts, found by the chat log's `api_prompt` and `extra_prompt`; one not found stays in `stable`."""
    extra = None
    if extra_prompt and prompt.endswith("\n" + extra_prompt):
        prompt, extra = prompt[: -len(extra_prompt) - 1], extra_prompt
    stable, time_line = _time_split(prompt, api_prompt)
    return Parts(stable, time_line, extra)


def _time_split(prompt: str, api_prompt: str | None) -> tuple[str, str | None]:
    if api_prompt:
        if prompt.endswith(api_prompt):
            return prompt, None
        if (at := prompt.rfind(api_prompt + "\n")) >= 0:
            end = at + len(api_prompt)
            return prompt[:end], prompt[end + 1 :]
    if found := _TIME_LINE.search(prompt):
        return prompt[: found.start()], found.group()[1:]
    return prompt, None
