"""What the two servers offer: their models, for the settings' pickers and for checking that a server answers."""

import asyncio
from typing import Any

import aiohttp
from yarl import URL

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import TIMEOUT_SECONDS


class ServerError(Exception):
    """The server did not list its models."""


def without_credentials(url: str) -> str:
    """The URL without its user and password, for what is shown or sent elsewhere."""
    return str(URL(url).with_user(None))


def decision_api_url(decider_url: str) -> str:
    """Where a decision server lists its models: the OpenAI-style `/v1` beside its `/v1/systemone`."""
    return decider_url.rstrip("/") + "/v1"


async def list_models(session: aiohttp.ClientSession, base_url: str) -> list[str]:
    """The model ids an OpenAI-compatible server offers (`/models`)."""
    return sorted(model["id"] for model in await _listed_models(session, base_url))


async def list_decision_models(session: aiohttp.ClientSession, base_url: str) -> list[str]:
    """The models a decision server offers.

    Those are the ones marked as decision models (output `decisions`, as llama-server and llama-swap report them), or
    every listed model when the server marks none.
    """
    models = await _listed_models(session, base_url)
    marked = [model for model in models if "decisions" in _output_modalities(model)]
    return sorted(model["id"] for model in marked or models)


async def models_or_empty(hass: HomeAssistant, base_url: str, *, decision: bool = False) -> list[str]:
    """The server's models for a picker; empty when it does not answer, since a name can still be typed."""
    listing = list_decision_models if decision else list_models
    try:
        return await listing(async_get_clientsession(hass), base_url)
    except ServerError:
        return []


async def _listed_models(session: aiohttp.ClientSession, base_url: str) -> list[dict[str, Any]]:
    url = base_url.rstrip("/") + "/models"
    try:
        async with asyncio.timeout(TIMEOUT_SECONDS), session.get(url) as resp:
            if resp.status != 200:
                raise ServerError(f"HTTP {resp.status} from {url}")
            data = await resp.json()
        return [model for model in data.get("data", []) if isinstance(model, dict) and model.get("id")]
    except (aiohttp.ClientError, TimeoutError, ValueError, AttributeError) as err:
        raise ServerError(f"{url}: {err}") from err


def _output_modalities(model: dict[str, Any]) -> list[str]:
    return (model.get("architecture") or {}).get("output_modalities") or []
