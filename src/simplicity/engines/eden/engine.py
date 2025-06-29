from dataclasses import dataclass, field
from typing import Literal, Self

from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent, AgentRunResult
from stone_brick.llm import (
    EndResult,
    TaskEventDeps,
    print_task_event,
)
from stone_brick.pydantic_ai_utils import (
    PydanticAIDeps,
    prod_run,
    prod_run_stream,
    with_events,
)

from simplicity.common.auto_translate import auto_translate
from simplicity.resources import (
    JinaClient,
    ModelWithSettings,
    Resource,
)
from simplicity.settings import Settings
from simplicity.structure import ReaderData, SearchData


class EdenEngineConfig(BaseModel):
    engine: Literal["eden"]
    agent_model_name: str
    translate_model_name: str


@dataclass(kw_only=True)
class AgentDeps(PydanticAIDeps):
    searched: dict[str, SearchData] = field(default_factory=dict)
    read: dict[str, ReaderData] = field(default_factory=dict)
    engine: "EdenEngine"


@with_events
async def search(ctx: RunContext[AgentDeps], query: str):
    """
    Search the web for relevant information.

    Args:
        query: The query to search for.
    """
    query_search = await auto_translate(
        ctx.deps.event_deps.spawn(),
        ctx.deps.engine.translate_model,
        query,
    )
    res = await ctx.deps.engine.jina_client.search(query_search)
    ctx.deps.searched.update({r.id_: r for r in res.data})
    return [r.llm_dump() for r in res.data]


@with_events
async def read(ctx: RunContext[AgentDeps], ids: list[str]):
    """
    Read the details of the search results.

    Args:
        ids: The IDs of the search results to read.
    """
    errors: list[str] = []
    correct_searched: list[str | SearchData] = []

    for id_ in ids:
        if id_ not in ctx.deps.searched:
            errors.append(id_)
        else:
            correct_searched.append(ctx.deps.searched[id_])

    newly = [
        r.data
        for r in await ctx.deps.engine.jina_client.read_batch(
            correct_searched, timeout=15
        )
        if not isinstance(r, Exception)
    ]
    ctx.deps.read.update({r.id_: r for r in newly})
    newly_dict = {r.id_: r.llm_dump() for r in newly}

    if errors:
        return (
            newly_dict,
            f"These IDs seems not found in previous searched results: {errors}",
        )
    return newly_dict


def submit():
    pass


general_agent = Agent(
    system_prompt="""You are an advanced research assistant specializing in comprehensive information retrieval and synthesis.

## Your Mission
Provide accurate, well-researched answers to user queries by systematically searching and analyzing web sources.

## Workflow Process
1. **Initial Search**: Use the search tool to find relevant sources for the query
2. **Source Evaluation**: Review search results and identify the most promising sources
3. **Deep Reading**: Use the read tool to extract detailed content from selected sources
4. **Iterative Research**: If the gathered information is insufficient:
   - Refine your search terms
   - Explore additional sources
   - Continue until you have comprehensive coverage of the topic

## Search Strategy Guidelines
- **Search Term Optimization**:
  - Use precise, relevant keywords
  - Consider synonyms and related terms
  - Adapt search strategy based on initial results
""",
    deps_type=AgentDeps,
    tools=[search, read],
    output_type=submit,
)


final_agent = Agent(
    system_prompt="""
You are a helpful research assistant that provides accurate answers based on the given information sources.

Instructions:
1. Answer the user's query using ONLY the information provided in the sources below
2. Include inline citations for all factual claims using the format: "Paris is the capital of France [source1 id][source2 id]..."
3. If you cannot find relevant information in the sources, clearly state that the information is not available in the provided sources
4. If the information sources are in different languages, respond in the same language as the user's query

Ensure your response is well-structured, accurate, properly cited, and as informative as possible by including relevant details from the sources.
    """,
    deps_type=PydanticAIDeps,
)


@dataclass
class EdenEngine:
    agent_model: ModelWithSettings
    translate_model: ModelWithSettings
    jina_client: JinaClient

    @classmethod
    def new(cls, settings: Settings, resource: Resource, engine_config: str) -> Self:
        try:
            eden_config = EdenEngineConfig.model_validate(
                settings.engine_configs[engine_config]
            )
        except KeyError:
            raise ValueError("Eden engine config not found") from None
        return cls(
            agent_model=resource.get_llm(eden_config.agent_model_name),
            translate_model=resource.get_llm(eden_config.translate_model_name),
            jina_client=resource.jina_client,
        )

    async def query(self, event_deps: TaskEventDeps, query: str):
        event_deps_0 = event_deps.spawn()
        agent_deps = AgentDeps(
            engine=self,
            event_deps=event_deps_0,
        )
        run = general_agent.run(
            model=self.agent_model.model,
            model_settings=self.agent_model.settings,
            user_prompt=f"<query>{query}</query>",
            deps=agent_deps,
        )
        _res: AgentRunResult[None] = await prod_run(event_deps_0, run)
        user_prompt = f"<informations>\n\n{[v.llm_dump() for v in agent_deps.read.values()]}\n\n</informations>\n\n<query>\n\n{query}\n\n</query>"

        final_run = final_agent.run_stream(
            model=self.agent_model.model,
            model_settings=self.agent_model.settings,
            user_prompt=user_prompt,
            deps=PydanticAIDeps(event_deps=event_deps),
        )
        final_res = await prod_run_stream(event_deps, final_run)
        return final_res.get_output()


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
        engine = EdenEngine.new(
            settings,
            resource,
            "eden",
        )
        event_deps = TaskEventDeps()
        cnt = 0
        async for event in event_deps.consume(
            lambda: engine.query(
                event_deps,
                "日本工签的种类及申请条件?",
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
