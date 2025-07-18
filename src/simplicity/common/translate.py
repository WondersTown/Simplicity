from pydantic import BaseModel
from pydantic_ai.agent import Agent
from stone_brick.pydantic_ai_utils import PydanticAIDeps

from simplicity.resources import ModelWithSettings
from simplicity.structure import SimpTaskDeps
from simplicity.utils import calc_usage


class Output(BaseModel):
    translation: str


agent = Agent(
    system_prompt="You are a professional translator. Provide only the translation wrapped by <result></result>, without any additional explanations or commentary. Preserve proper nouns, technical terms, and brand names in their original language. Maintain the original formatting and structure of the text.",
    deps_type=PydanticAIDeps,
    output_type=Output,
)


async def translate(
    deps: SimpTaskDeps, model: ModelWithSettings, query: str, lang: str
) -> str:
    """
    Translate text to a target language using the provided language model.

    Args:
        deps: Task event dependencies for tracking
        model: The language model to use for translation
        query: The text to translate
        lang: Target language code

    Returns:
        The translated text
    """

    res = await agent.run(
        model=model.model,
        model_settings=model.settings,
        user_prompt=f"<text>\n{query}\n</text>\n<target_lang>{lang}</target_lang>",
        deps=PydanticAIDeps(event_deps=deps),
    )
    usage = calc_usage(res.usage(), model.config_name)
    await deps.send(usage)
    return (res.output.translation + (f" lang:{lang}" if lang != "en" else ""))


if __name__ == "__main__":
    import logfire

    from simplicity.resources import Resource
    from simplicity.utils import get_settings_from_project_root

    logfire.configure()
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx()

    tasks = [("Hello, world!", "中文"), ("将本句话翻译成英语", "日本语")]

    settings = get_settings_from_project_root()
    resource = Resource(settings)
    model = resource.get_llm("google/gemini-2.5-flash-lite-preview-06-17")

    # async def main():
    #     results = await gather(
    #         *[translate(TaskEventDeps(), model, query, lang) for query, lang in tasks],
    #         batch_size=5,
    #     )
    #     print(results)

    # run(main)
