from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel
from pydantic_ai.agent import Agent

from simplicity.resources import (
    JinaReadClient,
    JinaSearchClient,
    ModelWithSettings,
    Resource,
    search_with_read,
)
from simplicity.settings import Settings
from simplicity.common.event import Event, print_event, EndResult, EventTaskOutput
from simplicity._pydantic_ai_adapter.agent_utils import BaseDeps, agent_run, fork_deps, prod_run, prod_run_stream, agent_run_stream
from stone_brick.observability import instrument
from stone_brick.asynclib import gather


class PardoEngineConfig(BaseModel):
    translate_model_name: str
    single_qa_model_name: str
    summary_qa_model_name: str


@dataclass
class PardoEngine:
    translate_llm: ModelWithSettings
    single_qa_llm: ModelWithSettings
    summary_qa_llm: ModelWithSettings
    jina_search_client: JinaSearchClient
    jina_read_client: JinaReadClient

    @classmethod
    def new(cls, settings: Settings, resource: Resource) -> Self:
        try:
            pardo_config = PardoEngineConfig.model_validate(
                settings.engine_configs["pardo"]
            )
        except KeyError:
            raise ValueError("Pardo engine config not found") from None
        if resource.jina_search_client is None or resource.jina_read_client is None:
            raise ValueError("Required Jina clients not found") from None
        return cls(
            translate_llm=resource.llms[pardo_config.translate_model_name],
            single_qa_llm=resource.llms[pardo_config.single_qa_model_name],
            summary_qa_llm=resource.llms[pardo_config.summary_qa_model_name],
            jina_search_client=resource.jina_search_client,
            jina_read_client=resource.jina_read_client,
        )

        
    async def _single_qa(self, deps: BaseDeps, query: str, source: str) -> str:
        SYSTEM_PROMPT = """
You are a helpful research assistant that provides accurate answers based on the given information sources.

Instructions:
1. Answer the user's query using ONLY the information provided in the sources below
2. If the information sources are in different languages, respond in the same language as the user's query

Ensure your response is well-structured, accurate, properly cited, and as informative as possible by including relevant details from the sources.
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
            deps_type=BaseDeps
        )
        run= agent.run(
            user_prompt,
            deps=deps,
        )
        res = await prod_run(deps, run)
        return res.output

    async def _translate(self, deps: BaseDeps, query: str, lang: str) -> str:
        agent = Agent(
            model=self.translate_llm.model,
            model_settings=self.translate_llm.settings,
            system_prompt="You are a professional translator. Provide only the translation wrapped by <result></result>, without any additional explanations or commentary. Preserve proper nouns, technical terms, and brand names in their original language. Maintain the original formatting and structure of the text.",
            deps_type=BaseDeps
        )
        run = agent.run(
            f"<text>\n{query}\n</text>\n<target_lang>{lang}</target_lang>",
            deps=deps
        )
        res = await prod_run(deps, run)
        res = res.output
        if res.startswith("<result>"):
            res = res[8:]  # Remove "<result>" from start
        if res.endswith("</result>"):
            res = res[:-9]  # Remove "</result>" from end
        res = res.strip()
        return res
    
    async def summary_qa(self, deps: BaseDeps | None, query: str, search_lang: str | None = "English"):
        deps = deps or BaseDeps()
        if search_lang is not None:
            query_search = await self._translate(fork_deps(deps), query, search_lang)
        else:
            query_search = query
        searched = await search_with_read(self.jina_search_client, self.jina_read_client, query_search, num=9, timeout=15)
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
            *[
                self._single_qa(fork_deps(deps), query, x.content)
                for x in searched
            ],
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
            deps_type=BaseDeps
        )
        forked_deps = fork_deps(deps)
        run = agent.run_stream(
            user_prompt,
            deps=forked_deps,
        )
        async for event in agent_run_stream(forked_deps, run):
            yield event

if __name__ == "__main__":
    from anyio import run
    from simplicity.settings import Settings
    from simplicity.resources import Resource
    from simplicity.utils import get_settings_from_project_root
    
    async def main():
        settings = get_settings_from_project_root()
        resource = Resource(settings)
        engine = PardoEngine.new(settings, resource)
        async for event in engine.summary_qa(None, "日本的首都是哪里?", "English"):
            if not isinstance(event, EndResult):
                print_event(event)

    run(main)