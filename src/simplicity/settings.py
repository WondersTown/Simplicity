from functools import cached_property
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator, Field
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings


class OAIProvider(BaseModel):
    base_url: str
    api_key: str

    def to_provider(self) -> OpenAIProvider:
        return OpenAIProvider(
            base_url=self.base_url,
            api_key=self.api_key,
        )


class OAILLMModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    provider: str
    settings: ModelSettings = Field(default_factory=ModelSettings)


class Settings(BaseModel):
    providers: dict[str, OAIProvider]
    llm_models: list[OAILLMModel]
    jina_api_key: str | None = None
    jina_reader_concurrency: int  = 3
    engine_configs: dict[str, dict[str, Any]]

    @model_validator(mode="after")
    def validate_provider(self) -> Self:
        for model in self.llm_models:
            if model.provider not in self.providers:
                raise ValueError(f"Provider {model.provider} not found")
        return self

    @cached_property
    def llm_models_by_name(self) -> dict[str, OAILLMModel]:
        return {model.name: model for model in self.llm_models}
