"""HA's play and volume tools with the player checked: they report success once the action is sent.

HA skips an unavailable player without an error and a player may not start or change its volume, so the model would
confirm what never happened. The checked play tool refuses an unavailable player and answers with what the player
then plays (media_player/llm.py); the checked volume tools fail when a player does not show the volume asked.
"""

from homeassistant.components.media_player import (
    ATTR_MEDIA_CONTENT_ID,
    MediaPlayerEntityFeature,
)
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent, llm
from homeassistant.util.json import JsonObjectType

from ..playback import async_started, now_playing, playback_of, unchanged
from ..volume import VOLUME_INTENTS, asked_by, async_volume_problem

PLAY_TOOL = "media_player__play_media"
# The players HA's play tool chooses from.
_PLAYS = MediaPlayerEntityFeature.SEARCH_MEDIA | MediaPlayerEntityFeature.PLAY_MEDIA


def checked(tool: llm.Tool) -> llm.Tool:
    """HA's play and volume tools checked; any other tool as it is."""
    if tool.name == PLAY_TOOL:
        return CheckedPlay(tool)
    if isinstance(tool, llm.IntentTool) and tool.intent_type in VOLUME_INTENTS:  # by intent: names carry a prefix
        return CheckedVolume(tool)
    return tool


def _seen_as(checked_tool: llm.Tool, tool: llm.Tool) -> None:
    """Give the checked tool HA's tool as the model sees it."""
    checked_tool.name = tool.name
    checked_tool.title = tool.title
    checked_tool.description = tool.description
    checked_tool.parameters = tool.parameters
    checked_tool.annotations = tool.annotations
    checked_tool.integration = tool.integration


class CheckedVolume(llm.Tool):
    """HA's volume tool, failed when a player does not show the volume asked."""

    def __init__(self, tool: llm.IntentTool) -> None:
        """Take HA's tool as the model sees it."""
        _seen_as(self, tool)
        self.tool = tool
        self.intent_type = tool.intent_type

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> llm.ToolResult | JsonObjectType:
        """Run HA's tool, then check its players (the intent's matched states); HomeAssistantError when one fails."""
        result = await self.tool.async_call(hass, tool_input, llm_context)
        response = getattr(result.data, "original", None) if isinstance(result, llm.ToolResult) else None
        if not isinstance(response, intent.IntentResponse):
            return result
        asked = asked_by(self.intent_type, {key: value for key, value in tool_input.tool_args.items()
                                     if not intent.is_blank_slot_value(value)})
        if asked is not None and (problem := await async_volume_problem(hass, response.matched_states, asked)):
            raise HomeAssistantError(problem)
        return result


class CheckedPlay(llm.Tool):
    """HA's play tool, its result what the player plays."""

    def __init__(self, tool: llm.Tool) -> None:
        """Take HA's tool as the model sees it."""
        _seen_as(self, tool)
        self.tool = tool

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> llm.ToolResult | JsonObjectType:
        """Play on the player HA's tool picks, once it is available; HomeAssistantError when nothing new plays.

        The arguments are read as HA's tool reads them: blank values, which models often send, are left out.
        """
        args = self.parameters({key: value for key, value in tool_input.tool_args.items()
                                if not intent.is_blank_slot_value(value)})
        player = _player(hass, llm_context, args)
        if player.state == STATE_UNAVAILABLE:
            raise HomeAssistantError(f"{player.name} is unavailable: nothing plays")
        before = playback_of(player)
        await self.tool.async_call(hass, tool_input, llm_context)
        if (started := await async_started(hass, player.entity_id, before)) is None:
            raise HomeAssistantError(f"{player.name} did not start playing")
        if unchanged(started, before) and started.attributes.get(ATTR_MEDIA_CONTENT_ID) != args[ATTR_MEDIA_CONTENT_ID]:
            raise HomeAssistantError(f"{player.name} still plays {now_playing(started) or 'what it played'}: "
                                     "the item did not start")
        return llm.ToolResult(data={"success": True, "now_playing": now_playing(started)})


def _player(hass: HomeAssistant, llm_context: llm.LLMContext, args: dict) -> State:
    """The one player the arguments name, matched as HA's play tool matches it."""
    constraints = intent.MatchTargetsConstraints(
        name=args.get("player_name"), area_name=args.get("player_area"), floor_name=args.get("player_floor"),
        domains={"media_player"}, assistant=llm_context.assistant, features=_PLAYS, single_target=True,
    )
    preferences = llm.async_get_match_preferences(hass, llm_context)
    result = intent.async_match_targets(hass, constraints, preferences)
    if not result.is_match:
        raise intent.MatchFailedError(result=result, constraints=constraints, preferences=preferences)
    return result.states[0]
