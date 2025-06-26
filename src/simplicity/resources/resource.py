from simplicity.resources.jina_client import JinaReadClient, JinaSearchClient
from simplicity.resources.pydantic_ai_llm import (
    ModelWithSettings,
    create_pydantic_ai_model,
)
from simplicity.settings import Settings


class Resource:
    llms: dict[str, ModelWithSettings]
    jina_search_client: JinaSearchClient | None = None
    jina_read_client: JinaReadClient | None = None

    def __init__(self, settings: Settings):
        self.llms = {}
        for model_name, model_config in settings.llm_models_by_name.items():
            self.llms[model_name] = ModelWithSettings(
                model=create_pydantic_ai_model(settings, model_name),
                settings=model_config.settings,
            )
        if settings.jina_api_key:
            self.jina_search_client = JinaSearchClient(settings.jina_api_key)
            self.jina_read_client = JinaReadClient(settings.jina_api_key)
