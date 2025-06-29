from typing import Any

from pydantic_ai.agent import Agent
from stone_brick.llm import TaskEventDeps
from stone_brick.pydantic_ai_utils import PydanticAIDeps, prod_run_stream

from simplicity.resources import ModelWithSettings


async def context_qa(
    deps: TaskEventDeps | None,
    llm: ModelWithSettings,
    query: str,
    contexts: list[Any],
):
    """
    Perform question-answering based on provided contexts.

    Args:
        deps: Task event dependencies for tracking (optional)
        llm: The language model to use for QA
        query: The user's question
        contexts: List of context strings to answer from

    Returns:
        The answer based on the provided contexts with citations
    """
    deps = deps or TaskEventDeps()

    SYSTEM_PROMPT = """
You are a research assistant that answers questions using only the provided information sources.

**Key Requirements:**
- Answer ONLY from the provided sources below
- Cite every factual claim inline: "fact [source_id]"
- State clearly if information is not available in sources
- Match the language of the user's query

Your response should be accurate, well-structured, and include all relevant details from the sources.
"""

    user_prompt = f"<informations>\n\n{contexts}\n\n</informations>\n\n<query>\n\n{query}\n\n</query>"

    agent = Agent(
        model=llm.model,
        model_settings=llm.settings,
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
    import logfire
    from anyio import run

    from simplicity.resources import Resource
    from simplicity.utils import get_settings_from_project_root

    logfire.configure()
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx()

    settings = get_settings_from_project_root()
    resource = Resource(settings)
    model = resource.get_llm("google/gemini-2.5-flash")
    contexts = [
        {
            "id": "22bf33",
            "content": "Paris is the capital of France",
        },
        {
            "id": "sa2d3fa",
            "content": "Vienna is the capital of Austria",
        },
        {
            "id": "320df83",
            "content": "Vichy was the capital of France during the second world war",
        },
        {
            "id": "sd3kw92",
            "content": "Château de Versailles was the home of Louis XIV",
        },
    ]

    async def main():
        res = await context_qa(None, model, "What is the capital of France?", contexts)
        print(res)

    run(main)
