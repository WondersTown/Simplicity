from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel
from pydantic_ai.agent import Agent
from stone_brick.asynclib import gather
from stone_brick.llm import (
    EndResult,
    EventTaskOutput,
    TaskEventDeps,
    print_task_event,
)
from stone_brick.observability import instrument
from stone_brick.pydantic_ai_utils import (
    PydanticAIDeps,
    prod_run,
    prod_run_stream,
)

from simplicity.common.auto_translate import auto_translate
from simplicity.resources import (
    JinaClient,
    ModelWithSettings,
    Resource,
)
from simplicity.settings import Settings


class PardoEngineConfig(BaseModel):
    engine: Literal["pardo"]
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
    def new(cls, settings: Settings, resource: Resource, engine_config: str) -> Self:
        try:
            pardo_config = PardoEngineConfig.model_validate(
                settings.engine_configs[engine_config]
            )
        except KeyError:
            raise ValueError("Pardo engine config not found") from None
        return cls(
            translate_llm=resource.get_llm(pardo_config.translate_model_name),
            single_qa_llm=resource.get_llm(pardo_config.single_qa_model_name),
            summary_qa_llm=resource.get_llm(pardo_config.summary_qa_model_name),
            jina_client=resource.jina_client,
        )

    async def _single_qa(self, deps: TaskEventDeps, query: str, source: str) -> str:
        SYSTEM_PROMPT = """
You are a helpful research assistant that provides accurate answers based on the given information sources.

Instructions:
1. Answer the user's query using ONLY the information provided in the sources below
2. If the information sources are in different languages, respond in the same language as the user's query

Ensure your response is well-structured, accurate, and as informative as possible by including relevant details from the sources.
"""

        user_prompt = f"""
<source>
{source}
</source>
<query>
{query}
</query>"""
        agent = Agent(
            model=self.single_qa_llm.model,
            model_settings=self.single_qa_llm.settings,
            system_prompt=SYSTEM_PROMPT,
            deps_type=PydanticAIDeps,
        )
        run = agent.run(
            user_prompt,
            deps=PydanticAIDeps(event_deps=deps),
        )
        res = await prod_run(deps, run)
        return res.output

    async def _translate(self, deps: TaskEventDeps, query: str, lang: str) -> str:
        agent = Agent(
            model=self.translate_llm.model,
            model_settings=self.translate_llm.settings,
            system_prompt="You are a professional translator. Provide only the translation wrapped by <result></result>, without any additional explanations or commentary. Preserve proper nouns, technical terms, and brand names in their original language. Maintain the original formatting and structure of the text.",
            deps_type=PydanticAIDeps,
        )
        run = agent.run(
            f"<text>\n{query}\n</text>\n<target_lang>{lang}</target_lang>",
            deps=PydanticAIDeps(event_deps=deps),
        )
        res = await prod_run(deps, run)
        res = res.output
        if res.startswith("<result>"):
            res = res[8:]  # Remove "<result>" from start
        if res.endswith("</result>"):
            res = res[:-9]  # Remove "</result>" from end
        res = res.strip()
        return res

    async def summary_qa(
        self,
        deps: TaskEventDeps | None,
        query: str,
        search_lang: str | Literal["auto"] | None = "auto",
    ):
        deps = deps or TaskEventDeps()
        if search_lang is None:
            query_search = query
        elif search_lang == "auto":
            query_search = await auto_translate(deps.spawn(), self.translate_llm, query)
        else:
            query_search = await self._translate(deps.spawn(), query, search_lang)
        searched = await self.jina_client.search_with_read(
            query_search,
            num=9,
            timeout=15,
        )
        await deps.event_send(EventTaskOutput(task_output=searched))
        SYSTEM_PROMPT = """
You are a helpful research assistant that provides accurate answers based on the given information sources.

Instructions:
1. Answer the user's query using ONLY the information provided in the sources below
2. Include inline citations for all factual claims using the format: "Paris is the capital of France [1][3][5]"
3. Use the source index numbers that correspond to the numbered information sources
4. If you cannot find relevant information in the sources, clearly state that the information is not available in the provided sources
5. If the information sources are in different languages, respond in the same language as the user's query

Ensure your response is well-structured, accurate, properly cited, and as informative as possible by including relevant details from the sources.
"""
        answers = await instrument(gather)(
            *[self._single_qa(deps.spawn(), query, x.content) for x in searched],
        )
        answers = [
            (
                ("published at " + searched[idx].publishedTime.isoformat() + "\n\n")  # type: ignore
                if searched[idx].publishedTime
                else ""
            )
            + x
            for idx, x in enumerate(answers)
            if not isinstance(x, Exception)
        ]

        texts_for_llm = "\n".join(
            [f"{idx + 1}.\n```\n{x}\n```" for idx, x in enumerate(answers)]
        )
        user_prompt = f"<informations>\n\n{texts_for_llm}\n\n</informations>\n\n<query>\n\n{query}\n\n</query>"
        agent = Agent(
            model=self.summary_qa_llm.model,
            model_settings=self.summary_qa_llm.settings,
            system_prompt=SYSTEM_PROMPT,
            deps_type=PydanticAIDeps,
        )
        run = agent.run_stream(
            user_prompt,
            deps=PydanticAIDeps(event_deps=deps),
        )
        res = await prod_run_stream(deps, run)
        return await res.get_output()


if __name__ == "__main__":
    from anyio import run

    from simplicity.resources import Resource
    from simplicity.settings import Settings
    from simplicity.utils import get_settings_from_project_root

    async def main():
        settings = get_settings_from_project_root()
        resource = Resource(settings)
        engine = PardoEngine.new(settings, resource, "pardo")
        cnt = 0
        event_deps = TaskEventDeps()
        async for event in event_deps.consume(
            lambda: engine.summary_qa(
                event_deps,
                "日本的首都是哪里?",
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
