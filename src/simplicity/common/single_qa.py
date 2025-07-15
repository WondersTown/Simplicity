from pydantic_ai.agent import Agent
from stone_brick.pydantic_ai_utils import PydanticAIDeps
from tiktoken import Encoding

from simplicity.resources import JinaClient, ModelWithSettings
from simplicity.structure import QAData, ReaderData, SearchData, SimpTaskDeps
from simplicity.utils import calc_usage

MIN_TOKEN_LENGTH = 400

async def single_qa_structured(
    deps: SimpTaskDeps,
    llm: ModelWithSettings,
    query: str,
    source: ReaderData | SearchData,
    *,
    jina: JinaClient | None = None,
    tokenizer: Encoding | None = None,
) -> QAData | None:
    if isinstance(source, SearchData):
        if jina is None:
            raise ValueError("Jina client is required for search data")
        source = (await jina.read(source)).data
    if tokenizer is not None:
        len_token = len(tokenizer.encode(source.content))
        if len_token < MIN_TOKEN_LENGTH:
            return None

    answer = await single_qa(deps, llm, query, source.content)
    return QAData(
        **source.model_dump(exclude={"kind"}),
        query=query,
        answer=answer,
    ) if answer is not None else None


async def single_qa(
    deps: SimpTaskDeps,
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
4. **Comprehensive Response**: Provide well-structured, informative answers. If a direct answer isn't possible, offer all relevant information from the source that relates to the query. Never include information that doesn't exist in the source.
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
    res = await agent.run(
        user_prompt,
        deps=PydanticAIDeps(event_deps=deps),
    )
    usage = calc_usage(res.usage(), llm.config_name)
    await deps.send(usage)
    res_output = res.output
    return res_output if res_output.strip() != "ERROR_PAGE" else None
