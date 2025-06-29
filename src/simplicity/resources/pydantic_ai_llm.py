from dataclasses import dataclass

from pydantic_ai.models import Model as PydanticAIModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

from simplicity.settings import Settings


@dataclass
class ModelWithSettings:
    model: PydanticAIModel
    settings: ModelSettings


def create_pydantic_ai_model(settings: Settings, model_conf_name: str) -> PydanticAIModel:
    try:
        model_config = settings.llm_configs[model_conf_name]
    except KeyError:
        raise ValueError(f"Model config {model_conf_name} not found in config") from None
    try:
        provider_config = settings.providers[model_config.provider]
    except KeyError:
        raise ValueError(
            f"Provider {model_config.provider} not found in config"
        ) from None
    return OpenAIModel(
        model_name=model_config.model_name,
        provider=provider_config.to_provider(),
    )
