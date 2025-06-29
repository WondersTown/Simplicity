from pydantic_ai.agent import Agent
from stone_brick.llm import TaskEventDeps
from stone_brick.pydantic_ai_utils import PydanticAIDeps, prod_run

from simplicity.resources import ModelWithSettings
from simplicity.resources.jina_client import ReaderData
from pydantic import BaseModel


class QAData(ReaderData):
    query: str
    answer: str

    def llm_dump(self) -> dict:
        return self.model_dump(
            exclude={
                "usage",
                "images",
                "links",
                "url",
                # For the raw reader data
                "title",
                "description",
                "content",
            }
        )


async def single_qa_structured(
    deps: TaskEventDeps,
    llm: ModelWithSettings,
    query: str,
    source: ReaderData,
) -> QAData:
    answer = await single_qa(deps, llm, query, source.content)
    return QAData(
        **source.model_dump(),
        query=query,
        answer=answer,
    )


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
You are a research assistant that answers questions based strictly on provided sources.

**Rules:**
1. Use ONLY information from the source material - no external knowledge
2. Match the language of the user's query
3. If information is missing, say so clearly
4. Structure answers clearly and concisely

**Format:**
- Start with a direct answer
- Support with source details
- State explicitly if the source lacks the requested information
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
