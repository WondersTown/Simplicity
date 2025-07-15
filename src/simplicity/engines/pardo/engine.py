from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel
from stone_brick.asynclib import gather
from stone_brick.asynclib.stream_runner import StreamRunner
from stone_brick.llm import (
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
from simplicity.structure import (
    InfoData,
    QAData,
    ReaderData,
    SearchData,
    SimpOutput,
    SimpTaskDeps,
    SimpTaskEvent,
)


class PardoEngineConfig(BaseModel):
    engine: Literal["pardo"] = "pardo"
    translate_model_name: str
    single_qa_model_name: str
    summary_qa_model_name: str
    read_pages: int

@dataclass
class PardoEngine:
    config: PardoEngineConfig
    translate_llm: ModelWithSettings
    single_qa_llm: ModelWithSettings
    summary_qa_llm: ModelWithSettings
    jina_client: JinaClient
    resource: Resource

    @classmethod
    def new(
        cls,
        settings: Settings,
        resource: Resource,
        engine_config: str | dict| PardoEngineConfig,
    ) -> Self:
        try:
            if isinstance(engine_config, (str, dict)):
                pardo_config = PardoEngineConfig.model_validate(
                    settings.engine_configs[engine_config] if isinstance(engine_config, str) else engine_config
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
            resource=resource,
        )

    async def _search(
        self,
        deps: SimpTaskDeps,
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
        searched = await self.jina_client.search(
            query_search,
            page=1,
        )
        await deps.send(TaskOutput(data=SimpOutput.gen(searched.data)))
        return searched.data

    async def _map_reduce_qa(
        self, deps: SimpTaskDeps, query: str, contexts: dict[str, ReaderData | SearchData], *, jina: JinaClient | None = None
    ):
        idx_contexts = list(contexts.values())
        answers = await instrument(gather)(
            *[
                single_qa_structured(deps.spawn(), self.single_qa_llm, query, x, jina=jina, tokenizer=self.resource.tokenizer)
                for x in idx_contexts
            ],
        )
        answers = [x for x in answers if x is not None and not isinstance(x, Exception)]
        await deps.send(TaskOutput(data=SimpOutput.gen(answers)))
        return answers

    @instrument
    async def summary_qa(
        self,
        deps: SimpTaskDeps,
        query: str,
        search_lang: str | Literal["auto"] | None = "auto",
    ):
        read = await self._search(deps.spawn(), query, search_lang)
        contexts = await self._map_reduce_qa(
            deps.spawn(), query, {x.id_: x for x in read}, jina=self.jina_client
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
    logfire.instrument_openai()
    logfire.instrument_httpx()
    logfire.instrument_pydantic_ai()
    

    async def main():
        settings = get_settings_from_project_root()
        resource = Resource(settings)
        engine = PardoEngine.new(settings, resource, "pardo-flash")
        cnt = 0
        collected: dict[str, ReaderData | SearchData | QAData] = {}
        with StreamRunner[SimpTaskEvent, str]() as runner:
            event_deps = SimpTaskDeps(producer=runner.producer)
            async with runner.run(
                engine.summary_qa(event_deps, "")
            ) as loop:
                async for event in loop:
                    if isinstance(event.content, TaskOutput) and isinstance(event.content.data, list):
                        for x in event.content.data:
                            if isinstance(x.d, InfoData):
                                collected[x.d.id_] = x.d
                    cnt += 1
                    print(f"thinking: {cnt}")
                    print_task_event(event)
            
            ans = runner.result
            # links = match_link(ans)
            
            print(f"result: {ans}")

    run(main)
