from functools import cached_property

from httpx import AsyncClient

from simplicity.resources.jina_client import JinaClient
from simplicity.resources.pydantic_ai_llm import (
    ModelWithSettings,
    create_pydantic_ai_model,
)
from simplicity.settings import Settings


class Resource:
    settings: Settings
    _llms: dict[str, ModelWithSettings]

    def __init__(self, settings: Settings):
        self.settings = settings
        # Initialize all LLM models
        self._llms = {
            model_name: ModelWithSettings(
                model=create_pydantic_ai_model(settings, model_name),
                settings=model_config.settings,
            )
            for model_name, model_config in settings.llm_models_by_name.items()
        }

    def get_llm(self, llm_name: str) -> ModelWithSettings:
        """
        Get an LLM model by name from the pre-initialized models.

        Args:
            llm_name: The name of the LLM model to retrieve

        Returns:
            ModelWithSettings instance for the requested model

        Raises:
            KeyError: If the model name is not found
        """
        if llm_name not in self._llms:
            if llm_name not in self.settings.llm_models_by_name:
                raise KeyError(f"LLM model '{llm_name}' not found in settings")
            self._llms[llm_name] = ModelWithSettings(
                model=create_pydantic_ai_model(self.settings, llm_name),
                settings=self.settings.llm_models_by_name[llm_name].settings,
            )
        return self._llms[llm_name]

    @cached_property
    def http_client(self) -> AsyncClient:
        """Cached HTTP client. Created once on first access."""
        return AsyncClient()

    @cached_property
    def jina_client(self) -> JinaClient:
        """Cached Jina client. Created once on first access."""
        if not self.settings.jina_api_key:
            raise ValueError("Jina API key is not set")

        return JinaClient(api_key=self.settings.jina_api_key, client=self.http_client)

    async def close(self):
        """Close the HTTP client when done."""
        await self.http_client.aclose()
