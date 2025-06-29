from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel
from stone_brick.asynclib import gather
from stone_brick.llm import (
    EndResult,
    TaskEventDeps,
    print_task_event,
)
from stone_brick.observability import instrument

from simplicity.common.auto_translate import auto_translate
from simplicity.common.context_qa import context_qa
from simplicity.common.split_question import (
    splitting_question,
)
from simplicity.common.translate import translate
from simplicity.engines.pardo.engine import PardoEngine, PardoEngineConfig
from simplicity.resources import (
    JinaClient,
    ModelWithSettings,
    Resource,
)
from simplicity.settings import Settings


class VillVEngineConfig(BaseModel):
    engine: Literal["villv"] = "villv"
    recursive_splitting: bool = False
    translate_model_name: str
    split_model_name: str
    qa_model_name: str
    summary_model_name: str


@dataclass
class VillVEngine:
    pardo_engine: PardoEngine
    jina_client: JinaClient
    recursive_splitting: bool
    split_model: ModelWithSettings

    @classmethod
    def new(cls, settings: Settings, resource: Resource, engine_config: str) -> Self:
        try:
            villv_config = VillVEngineConfig.model_validate(
                settings.engine_configs[engine_config]
            )
        except KeyError:
            raise ValueError("VillV engine config not found") from None
        return cls(
            pardo_engine=PardoEngine.new(
                settings,
                resource,
                PardoEngineConfig(
                    engine="pardo",
                    translate_model_name=villv_config.translate_model_name,
                    single_qa_model_name=villv_config.qa_model_name,
                    summary_qa_model_name=villv_config.summary_model_name,
                ),
            ),
            jina_client=resource.jina_client,
            recursive_splitting=villv_config.recursive_splitting,
            split_model=resource.get_llm(villv_config.split_model_name),
        )

    async def query(
        self,
        deps: TaskEventDeps | None,
        query: str,
        search_lang: str | Literal["auto"] | None = "auto",
    ):
        deps = deps or TaskEventDeps()
        return await self.normal_query(
            deps, query, search_lang, recursive=self.recursive_splitting
        )

    @instrument
    async def normal_query(
        self,
        deps: TaskEventDeps,
        query: str,
        search_lang: str | Literal["auto"] | None = "auto",
        recursive: bool = False,
    ):
        if search_lang == "auto":
            search_query = await auto_translate(
                deps.spawn(), self.pardo_engine.translate_llm, query
            )
        elif search_lang is not None:
            search_query = await translate(
                deps.spawn(), self.pardo_engine.translate_llm, query, search_lang
            )
        else:
            search_query = query

        subqueries = await splitting_question(
            deps.spawn(), self.split_model, search_query
        )

        if len(subqueries) == 1:
            return await self.pardo_engine.summary_qa(deps, query, None)

        answers = await instrument(gather)(
            *(
                [
                    self.pardo_engine.summary_qa(deps.spawn(), query, None)
                    for query in subqueries
                ]
                if not recursive
                else [
                    self.normal_query(deps.spawn(), query, None, recursive=True)
                    for query in subqueries
                ]
            ),
        )
        answers = [x for x in answers if not isinstance(x, Exception)]
        if len(answers) == 1:
            return answers[0]

        return await context_qa(deps, self.pardo_engine.summary_qa_llm, query, answers)


async def main():
    pass


if __name__ == "__main__":
    import logfire
    from anyio import run

    from simplicity.resources import Resource
    from simplicity.settings import Settings
    from simplicity.utils import get_settings_from_project_root

    logfire.configure()
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx()

    async def main():
        settings = get_settings_from_project_root()
        resource = Resource(settings)
        engine = VillVEngine.new(settings, resource, "villv-pro")
        cnt = 0
        event_deps = TaskEventDeps()
        async for event in event_deps.consume(
            lambda: engine.query(
                event_deps,
                "比较新加坡、香港的经济政策和住房负担能力，以及这些因素对人才吸引力的影响",
            )
        ):
            cnt += 1
            if not isinstance(event, EndResult):
                if event[1]:
                    print_task_event(event[0])
                else:
                    print(f"thinking: {cnt}")
                    print_task_event(event[0])

    run(main)
