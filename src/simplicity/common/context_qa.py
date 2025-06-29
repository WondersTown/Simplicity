from pydantic_ai.agent import Agent
from stone_brick.llm import TaskEventDeps
from stone_brick.pydantic_ai_utils import PydanticAIDeps, prod_run_stream

from simplicity.resources import ModelWithSettings


async def context_qa(
    deps: TaskEventDeps | None,
    llm: ModelWithSettings,
    query: str,
    contexts: list[str],
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
You are a helpful research assistant that provides accurate answers based on the given information sources.

Instructions:
1. Answer the user's query using ONLY the information provided in the sources below
2. Include inline citations for all factual claims using the format: "Paris is the capital of France [1][3][5]"
3. Use the source index numbers that correspond to the numbered information sources
4. If you cannot find relevant information in the sources, clearly state that the information is not available in the provided sources
5. If the information sources are in different languages, respond in the same language as the user's query

Ensure your response is well-structured, accurate, properly cited, and as informative as possible by including relevant details from the sources.
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