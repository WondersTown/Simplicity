from dataclasses import dataclass, field
from functools import cached_property

from httpx import AsyncClient

from simplicity.resources.jina_client import JinaClient
from simplicity.resources.pydantic_ai_llm import (
    ModelWithSettings,
    create_pydantic_ai_model,
)
from simplicity.settings import Settings


@dataclass
class Resource:
    settings: Settings
    _llms: dict[str, ModelWithSettings] = field(default_factory=dict)

    def get_llm(self, llm_conf_name: str) -> ModelWithSettings:
        """
        Get an LLM model by name from the pre-initialized models.

        Args:
            llm_name: The name of the LLM model to retrieve

        Returns:
            ModelWithSettings instance for the requested model

        Raises:
            KeyError: If the model name is not found
        """
        if llm_conf_name not in self._llms:
            if llm_conf_name not in self.settings.llm_configs:
                raise KeyError(f"LLM model '{llm_conf_name}' not found in settings")
            self._llms[llm_conf_name] = ModelWithSettings(
                model=create_pydantic_ai_model(self.settings, llm_conf_name),
                settings=self.settings.llm_configs[llm_conf_name].settings,
            )
        return self._llms[llm_conf_name]

    @cached_property
    def http_client(self) -> AsyncClient:
        """Cached HTTP client. Created once on first access."""
        return AsyncClient()

    @cached_property
    def jina_client(self) -> JinaClient:
        """Cached Jina client. Created once on first access."""
        if not self.settings.jina_api_key:
            raise ValueError("Jina API key is not set")

        return JinaClient(
            api_key=self.settings.jina_api_key,
            client=self.http_client,
            concurrency=self.settings.jina_reader_concurrency,
        )

    async def close(self):
        """Close the HTTP client when done."""
        await self.http_client.aclose()
