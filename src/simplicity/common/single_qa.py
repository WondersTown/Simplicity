from pydantic import BaseModel
from pydantic_ai.agent import Agent
from stone_brick.llm import TaskEventDeps
from stone_brick.pydantic_ai_utils import PydanticAIDeps, prod_run

from simplicity.resources import ModelWithSettings
from simplicity.structure import QAData, ReaderData


async def single_qa_structured(
    deps: TaskEventDeps,
    llm: ModelWithSettings,
    query: str,
    source: ReaderData,
) -> QAData | None:
    answer = await single_qa(deps, llm, query, source.content)
    return QAData(
        **source.model_dump(),
        query=query,
        answer=answer,
    ) if answer is not None else None


async def single_qa(
    deps: TaskEventDeps,
    llm: ModelWithSettings,
    query: str,
    source: str,
) -> str | None:
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
You are a professional research assistant specializing in source-based question answering.

**Core Principles:**
1. **Source Fidelity**: Answer exclusively using information from the provided source material. Do not incorporate external knowledge or assumptions.
2. **Language Matching**: Respond in the same language as the user's query to ensure accessibility and clarity.
3. **Transparency**: When information is incomplete or unavailable in the source, explicitly state this limitation.
4. **Comprehensive Response**: Provide well-structured, informative answers. If a direct answer isn't possible, offer all relevant information from the source that relates to the query.
"""
# **Special Handling:**
# If the provided source appears to be an error page or contains no actual informational content, respond with exactly: `ERROR_PAGE`
# """

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
    return res.output if res.output.strip() != "ERROR_PAGE" else None
