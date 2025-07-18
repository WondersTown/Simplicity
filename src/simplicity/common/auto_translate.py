import logging
from typing import Any

from pydantic import BaseModel
from pydantic_ai.agent import Agent
from stone_brick.llm import (
    TaskEventDeps,
)
from stone_brick.pydantic_ai_utils import (
    PydanticAIDeps,
)

from simplicity.resources import (
    ModelWithSettings,
    Resource,
)
from simplicity.structure import SimpTaskDeps
from simplicity.utils import calc_usage

logger = logging.getLogger(__name__)


class Output(BaseModel):
    origin_lang: str
    target_lang: str
    translated_query: str


agent = Agent(
    system_prompt="""You are a translation assistant that determines the appropriate target language for queries.

Target language selection:
- Default: English
- Use local language when query involves:
  • Location-specific services or systems
  • Cultural or traditional practices
  • Official processes or regulations of a specific country
  • Local market information or businesses
  • Region-specific current events

Stay in English for:
  • Academic research, papers, or scientific content
  • International business or technology topics
  • General knowledge questions
  • Global products or services

Examples:
- "How to take subway in Shanghai?" → zh (local transportation system)
- "日本的工签政策及具体要求" → ja (Japan-specific regulations)
- "人工智能的最新发展" → en (general technology topic)
- "How to apply for a work visa in Germany?" → de (German official process)
- "Weather in Paris" → en (general information query)
- "Research papers by 大连理工大学" → en (academic content is international)

Key principles:
- If the answer would differ significantly by country/region, use local language
- For regions with multiple languages, prefer the most popular language
- The original language of the query is not important, do not consider it

Rules:
- The translated text should be clear enough to indicate the real intention of the original query
- The output query should be optimized for search engine, not necessarily excatly the same as the original query
- Preserve proper nouns, technical terms, and brand names
- Maintain original formatting and structure""",
    deps_type=PydanticAIDeps,
    output_type=Output,
)


async def _auto_translate(deps: SimpTaskDeps, model: ModelWithSettings, query: str):
    res = await agent.run(
        model=model.model,
        model_settings=model.settings,
        user_prompt=f"<query>\n{query}\n</query>",
        deps=PydanticAIDeps(event_deps=deps),
        output_type=Output,
    )
    usage = calc_usage(res.usage(), model.config_name)
    await deps.send(usage)
    return res.output


async def auto_translate(deps: SimpTaskDeps, model: ModelWithSettings, query: str):
    res = await _auto_translate(deps, model, query)
    return (res.translated_query + (f" lang:{res.target_lang}" if res.target_lang != "en" else ""))


if __name__ == "__main__":
    import logfire
    from anyio import run
    from stone_brick.asynclib import gather
    from stone_brick.asynclib.stream_runner import StreamRunner

    from simplicity.resources import Resource
    from simplicity.utils import get_settings_from_project_root

    logfire.configure()
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx()

    tasks = [
        ("新加坡地铁系统地图和票价计算", ("en", "zh")),  # Multi-lingual country
        (
            "英国脱欧对英国移民政策的影响",
            ("en",),
        ),  # Political topic, not UK-specific service
        ("如何在硅谷注册创业公司", ("en",)),  # US location but international audience
        ("京都传统茶道", ("ja",)),  # Cultural practice
        ("Python编程教程", ("en",)),  # Programming is universal
        ("孟买本地火车时刻表", ("hi", "en", "mr")),  # India has multiple languages
        (
            "瑞士银行对外国人的监管规定",
            ("en", "de", "fr", "it"),
        ),  # Multi-lingual country
        ("曼谷米其林星级餐厅", ("th", "en")),  # International rating system
        ("多伦多的中医执业者", ("en",)),  # Chinese topic but in Canada
        ("魁北克的中医执业者", ("fr",)),  # Chinese topic but in Canada
        ("2026年FIFA世界杯门票", ("en",)),  # International event
        (
            "柏林创业生态系统概览",
            ("en", "de"),
        ),  # Tech ecosystem - international or local?
        ("魁北克法语与法国法语的区别", ("fr", "en")),  # Language comparison
    ]

    settings = get_settings_from_project_root()
    resource = Resource(settings)
    model = resource.get_llm("deepseek-v3")

    async def main():
        # Run all translations in parallel
        with StreamRunner[Any, list[Output | Exception]]() as runner:
            async with runner.run(
        gather(
                *[_auto_translate(TaskEventDeps(producer=runner.producer), model, query) for query, _ in tasks],
                batch_size=5,
            ))as loop:
                async for event in loop:
                    pass
            results = runner.result

        # Check results
        failed = []
        for (query, acceptable_langs), res in zip(tasks, results, strict=False):
            if isinstance(res, Exception):
                failed.append((query, "Exception", acceptable_langs, "Exception"))
                continue
            is_correct = res.target_lang in acceptable_langs
            if not is_correct:
                failed.append(
                    (query, res.target_lang, acceptable_langs, res.translated_query)
                )

        if failed:
            print(f"Failed {len(failed)} out of {len(tasks)} tests:\n")
            for query, detected, acceptable, translated in failed:
                print(f"✗ Query: {query}")
                print(f"  Target: {detected} (expected: {', '.join(acceptable)})")
                print(f"  Translated: {translated}")
                print()
        else:
            print(f"✓ All {len(tasks)} tests passed!\n")
            for (query, acceptable_langs), res in zip(tasks, results, strict=False):
                if isinstance(res, Exception):
                    continue
                print(f"✓ Query: {query}")
                print(
                    f"  Target: {res.target_lang} (acceptable: {', '.join(acceptable_langs)})"
                )
                print(f"  Translated: {res.translated_query}")
                print()

    run(main)
