from dataclasses import dataclass
from typing import Literal, Self, Sequence, TypeAlias

from pydantic import BaseModel
from stone_brick.asynclib import gather
from stone_brick.asynclib.stream_runner import StreamRunner
from stone_brick.llm import (
    TaskEvent,
    TaskEventDeps,
    TaskOutput,
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
from simplicity.structure import InfoData, QAData, ReaderData, SearchData
from simplicity.utils import match_link


class PardoEngineConfig(BaseModel):
    engine: Literal["pardo"] = "pardo"
    translate_model_name: str
    single_qa_model_name: str
    summary_qa_model_name: str
    read_pages: int

OutputDataType: TypeAlias = Sequence[InfoData] | str

@dataclass
class PardoEngine:
    config: PardoEngineConfig
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
            config=pardo_config,
            translate_llm=resource.get_llm(pardo_config.translate_model_name),
            single_qa_llm=resource.get_llm(pardo_config.single_qa_model_name),
            summary_qa_llm=resource.get_llm(pardo_config.summary_qa_model_name),
            jina_client=resource.jina_client,
        )

    async def _search(
        self,
        deps: TaskEventDeps[OutputDataType],
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
            num=self.config.read_pages,
        )
        await deps.send(TaskOutput(data=searched))
        return searched

    async def _map_reduce_qa(
        self, deps: TaskEventDeps[OutputDataType], query: str, contexts: dict[str, ReaderData]
    ):
        idx_contexts = list(contexts.values())
        answers = await instrument(gather)(
            *[
                single_qa_structured(deps.spawn(), self.single_qa_llm, query, x)
                for x in idx_contexts
            ],
        )
        answers = [x for x in answers if not isinstance(x, Exception)]
        await deps.send(TaskOutput(data=answers))
        return answers

    @instrument
    async def summary_qa(
        self,
        deps: TaskEventDeps[OutputDataType],
        query: str,
        search_lang: str | Literal["auto"] | None = "auto",
    ):
        read = await self._search(deps.spawn(), query, search_lang)
        contexts = await self._map_reduce_qa(
            deps.spawn(), query, {str(x.id_): x for x in read}
        )
        llm_contexts = [x.llm_dump() for x in contexts]
        return await context_qa(deps, self.summary_qa_llm, query, llm_contexts)


if __name__ == "__main__":
    import logfire
    from anyio import run

    from simplicity.resources import Resource
    from simplicity.settings import Settings
    from simplicity.utils import get_settings_from_project_root
    logfire.configure()
    logfire.instrument_httpx()
    logfire.instrument_pydantic_ai()
    

    async def main():
        settings = get_settings_from_project_root()
        resource = Resource(settings)
        engine = PardoEngine.new(settings, resource, "pardo-flash")
        cnt = 0
        collected: dict[str, ReaderData | SearchData | QAData] = {}
        with StreamRunner[TaskEvent[OutputDataType], str]() as runner:
            event_deps = TaskEventDeps(producer=runner.producer)
            async with runner.run(
                engine.summary_qa(event_deps, "销售税应该向买家所在地缴纳还是卖家所在地缴纳?" )
            ) as loop:
                async for event in loop:
                    if isinstance(event.content, TaskOutput) and isinstance(event.content.data, list):
                        for x in event.content.data:
                            if isinstance(x, (ReaderData, SearchData, QAData)):
                                collected[x.id_] = x
                    cnt += 1
                    print(f"thinking: {cnt}")
                    print_task_event(event)
            
            ans = runner.result
            # links = match_link(ans)
            
            print(f"result: {ans}")

    run(main)
