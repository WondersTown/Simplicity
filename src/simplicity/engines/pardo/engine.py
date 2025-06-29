from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel
from stone_brick.asynclib import gather
from stone_brick.llm import (
    EndResult,
    EventTaskOutput,
    TaskEventDeps,
    print_task_event,
)
from stone_brick.observability import instrument

from simplicity.common.auto_translate import auto_translate
from simplicity.common.context_qa import context_qa
from simplicity.common.single_qa import single_qa_structured
from simplicity.common.translate import translate
from simplicity.resources import (
    JinaClient,
    ModelWithSettings,
    Resource,
)
from simplicity.settings import Settings
from simplicity.structure import ReaderData


class PardoEngineConfig(BaseModel):
    engine: Literal["pardo"] = "pardo"
    translate_model_name: str
    single_qa_model_name: str
    summary_qa_model_name: str


@dataclass
class PardoEngine:
    translate_llm: ModelWithSettings
    single_qa_llm: ModelWithSettings
    summary_qa_llm: ModelWithSettings
    jina_client: JinaClient

    @classmethod
    def new(
        cls,
        settings: Settings,
        resource: Resource,
        engine_config: str | PardoEngineConfig,
    ) -> Self:
        try:
            if isinstance(engine_config, str):
                pardo_config = PardoEngineConfig.model_validate(
                    settings.engine_configs[engine_config]
                )
            else:
                pardo_config = engine_config
        except KeyError:
            raise ValueError("Pardo engine config not found") from None
        return cls(
            translate_llm=resource.get_llm(pardo_config.translate_model_name),
            single_qa_llm=resource.get_llm(pardo_config.single_qa_model_name),
            summary_qa_llm=resource.get_llm(pardo_config.summary_qa_model_name),
            jina_client=resource.jina_client,
        )

    async def _search(
        self,
        deps: TaskEventDeps,
        query: str,
        search_lang: str | Literal["auto"] | None = "auto",
    ):
        if search_lang is None:
            query_search = query
        elif search_lang == "auto":
            query_search = await auto_translate(deps.spawn(), self.translate_llm, query)
        else:
            query_search = await translate(
                deps.spawn(), self.translate_llm, query, search_lang
            )
        searched = await self.jina_client.search_with_read(
            query_search,
            num=9,
            timeout=15,
        )
        await deps.event_send(EventTaskOutput(task_output=searched))
        return searched

    async def _map_reduce_qa(
        self, deps: TaskEventDeps, query: str, contexts: dict[str, ReaderData]
    ):
        idx_contexts = list(contexts.values())
        answers = await instrument(gather)(
            *[
                single_qa_structured(deps.spawn(), self.single_qa_llm, query, x)
                for x in idx_contexts
            ],
        )
        return [x for x in answers if not isinstance(x, Exception)]

    async def summary_qa(
        self,
        deps: TaskEventDeps | None,
        query: str,
        search_lang: str | Literal["auto"] | None = "auto",
    ):
        deps = deps or TaskEventDeps()
        read = await self._search(deps.spawn(), query, search_lang)
        contexts = await self._map_reduce_qa(
            deps.spawn(), query, {str(x.id_): x for x in read}
        )
        llm_contexts = [x.llm_dump() for x in contexts]
        return await context_qa(deps, self.summary_qa_llm, query, llm_contexts)


if __name__ == "__main__":
    from anyio import run

    from simplicity.resources import Resource
    from simplicity.settings import Settings
    from simplicity.utils import get_settings_from_project_root

    async def main():
        settings = get_settings_from_project_root()
        resource = Resource(settings)
        engine = PardoEngine.new(settings, resource, "pardo-pro")
        cnt = 0
        event_deps = TaskEventDeps()
        async for event in event_deps.consume(
            lambda: engine.summary_qa(event_deps, "日本的首都是哪里?", "日本语")
        ):
            cnt += 1
            if not isinstance(event, EndResult):
                if event[1]:
                    print_task_event(event[0])
                else:
                    print(f"thinking: {cnt}")
                    print_task_event(event[0])

    run(main)
