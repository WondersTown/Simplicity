from typing import Any

from pydantic_ai.agent import Agent
from stone_brick.pydantic_ai_utils import PydanticAIDeps, prod_run_stream

from simplicity.resources import ModelWithSettings
from simplicity.structure import SimpTaskDeps
from simplicity.utils import calc_usage


async def context_qa(
    deps: SimpTaskDeps,
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
You are an expert research assistant that provides accurate, well-sourced answers to user questions.

**Core Principles:**
- Use your professional knowledge and expertise to construct comprehensive, insightful answers
- Leverage provided sources as evidence and support, but don't limit yourself to merely summarizing them
- Synthesize information from sources with your domain expertise to provide deeper understanding
- Maintain objectivity and accuracy while offering informed analysis and interpretation
- Adapt your communication style to match the user's language and context

**Response Structure:**

1. **Summary**: Begin with "TL;DR: ..." providing a concise one-line answer
2. **Detailed Analysis**: Follow with a comprehensive explanation that:
   - Constructs a thoughtful, expert-level response using your professional knowledge
   - Integrates source information with analytical insights and broader context
   - Presents information as a coherent narrative rather than disconnected points
   - Establishes context and explains relationships between concepts
   - Addresses implications and broader significance when relevant
   - Uses clear, accessible language appropriate for the topic complexity

**Source Management:**

1. **Citations**: Reference sources immediately after each claim using their unique identifiers, which is like regex [a-f0-9]{6}
   - Example: "The population reached 2.1 million [a1b2c3, d4e5f6]"
   - Use source IDs, not URLs, for citations
   - When multiple sources contain similar information, cite only the most authoritative or comprehensive ones to avoid redundancy

2. **Source Evaluation**: Prioritize information from authoritative sources (official websites, mainstream media, established institutions, peer-reviewed publications)

3. **Information Gaps**: When sources lack specific information, draw upon your professional expertise: "While the sources don't address X, based on established knowledge in this field..." or "From a professional perspective, this typically means..."

**Quality Guidelines:**
- Apply critical thinking and professional judgment to construct meaningful answers
- Go beyond source compilation to provide expert analysis and interpretation
- Filter out irrelevant information that doesn't serve the user's question
- Ensure logical flow and coherent argumentation
- Match the user's language preference
- Provide sufficient detail for understanding without overwhelming
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
    usage = calc_usage(res.usage(), llm.config_name)
    await deps.send(usage)
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
