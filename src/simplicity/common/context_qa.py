from typing import Any

from pydantic_ai.agent import Agent
from stone_brick.llm import TaskEventDeps
from stone_brick.pydantic_ai_utils import PydanticAIDeps, prod_run_stream

from simplicity.resources import ModelWithSettings


async def context_qa(
    deps: TaskEventDeps,
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
    SYSTEM_PROMPT = """
You are a research assistant that answers questions using the provided sources.

**Instructions:**

1. **Citations**: 
   - Cite every fact immediately after stating it: "Paris is the capital [22bf33]" or "This is documented [22bf33, a32d83]"
   - Some information may be irrelevant to the question, simply ignore them

2. **Language Matching**: Respond in the same language as the user's question

3. **Answers**: 
   - Start with like "TL;DR: ..." to give the answer in a short summary, do not repeat the question
   - Give direct, substantive, well-structured and easy-to-understand answers with relevant details from the sources. 
   - If some information is missing, clearly state what's not available in the sources, but try to answer the question as much as you can. It would be like "Although the information is not available in the sources, my knowledge tells me that..."
   - Use conversational, natural language that matches the query's tone, avoid overly formal or academic language unless the query requires it
   - Be concise and focus on what the user actually wants to know
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

    # async def main():
    #     res = await context_qa(None, model, "What is the capital of France?", contexts)
    #     print(res)

    # run(main)
