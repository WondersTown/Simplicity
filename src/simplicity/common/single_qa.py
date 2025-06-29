from pydantic_ai.agent import Agent
from stone_brick.llm import TaskEventDeps
from stone_brick.pydantic_ai_utils import PydanticAIDeps, prod_run

from simplicity.resources import ModelWithSettings


async def single_qa(
    deps: TaskEventDeps,
    llm: ModelWithSettings,
    query: str,
    source: str,
) -> str:
    """
    Perform single question-answering based on a given source.
    
    Args:
        deps: Task event dependencies for tracking
        llm: The language model to use for QA
        query: The user's question
        source: The source content to answer from
        
    Returns:
        The answer based on the source content
    """
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
        model=llm.model,
        model_settings=llm.settings,
        system_prompt=SYSTEM_PROMPT,
        deps_type=PydanticAIDeps,
    )
    run = agent.run(
        user_prompt,
        deps=PydanticAIDeps(event_deps=deps),
    )
    res = await prod_run(deps, run)
    return res.output 