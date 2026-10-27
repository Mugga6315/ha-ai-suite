"""AI Suite's actions: run one of Second Brain's nightly passes now."""

from functools import partial
from typing import TYPE_CHECKING

from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
    callback,
)
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError

from .brain.background import RunFailed
from .const import DOMAIN

if TYPE_CHECKING:
    from .brain.setup import Brain

SERVICE_CONSOLIDATE = "consolidate"
SERVICE_LEARN = "learn"
# The action, the Brain field holding its job, and the feature that adds the job (named in the error).
_JOBS = {SERVICE_CONSOLIDATE: ("consolidator", "Librarian"), SERVICE_LEARN: ("learner", "Self-improvement")}


@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Register the actions once, for as long as AI Suite is loaded."""
    for service in _JOBS:
        hass.services.async_register(
            DOMAIN, service, partial(_async_run_job, hass, service), supports_response=SupportsResponse.OPTIONAL
        )


async def _async_run_job(hass: HomeAssistant, service: str, _call: ServiceCall) -> ServiceResponse:
    """Run the job and return its summary; an error when it cannot run or changed nothing."""
    field, feature = _JOBS[service]
    job = getattr(_loaded_brain(hass, feature), field)
    if job is None:
        raise ServiceValidationError(
            translation_domain=DOMAIN, translation_key="feature_missing", translation_placeholders={"feature": feature}
        )
    try:
        summary = await job.async_run()
    except RunFailed as err:
        raise HomeAssistantError(
            translation_domain=DOMAIN, translation_key="run_failed", translation_placeholders={"reason": str(err)}
        ) from err
    return {"result": summary}


def _loaded_brain(hass: HomeAssistant, feature: str) -> Brain:
    entries = hass.config_entries.async_loaded_entries(DOMAIN)
    if not entries:
        raise ServiceValidationError(translation_domain=DOMAIN, translation_key="not_loaded")
    brain = entries[0].runtime_data.brain
    if brain is None:
        raise ServiceValidationError(
            translation_domain=DOMAIN, translation_key="feature_missing", translation_placeholders={"feature": feature}
        )
    return brain
