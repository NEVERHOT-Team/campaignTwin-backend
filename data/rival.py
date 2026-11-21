"""Return the demo's fixed rival scenarios until the evidence model is ready."""

from contracts.models import CampaignSpec, ContextPack, RivalScenario
from mocks import load_mock


def get_rival_scenarios(spec: CampaignSpec) -> list[RivalScenario]:
    return ContextPack.model_validate(load_mock("context_pack")).rivals
