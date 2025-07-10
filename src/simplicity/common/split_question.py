from pydantic import BaseModel
from pydantic_ai.agent import Agent
from stone_brick.llm import (
    TaskEventDeps,
)
from stone_brick.observability import instrument
from stone_brick.pydantic_ai_utils import (
    PydanticAIDeps,
    prod_run,
)

from simplicity.resources import (
    ModelWithSettings,
    Resource,
)


class Output(BaseModel):
    subqueries: list[str]


agent = Agent(
    system_prompt="""Split complex queries into sub-queries optimized for web search.

Core principles:
- Only split when necessary - prefer keeping related concepts together
- Each subquery must be self-contained and effective for web search
- Maintain the original language - never translate
- Focus on the core question, not just the presence of multiple clauses

DO NOT split when:
- Query has background context + single question: "虽然我已经学了三年编程，但是我想知道如何提高代码质量"
- Multiple aspects of the same topic: "ChatGPT的工作原理、优势和局限性"
- Single coherent topic: "解释量子计算的基本原理以及它与经典计算的区别"

DO split when:
- Genuinely distinct topics: "比较新加坡、香港和东京的经济政策，同时分析它们的公共交通系统和住房负担能力" → economic policies | public transport | housing
- Different service types: "我需要孟买本地火车时刻表、附近的街头小吃推荐，以及给游客的安全提示" → transport | food | safety
""",
    deps_type=PydanticAIDeps,
    output_type=Output,
)


async def splitting_question(deps: TaskEventDeps, model: ModelWithSettings, query: str):
    run = agent.run(
        model=model.model,
        model_settings=model.settings,
        user_prompt=f"<query>\n{query}\n</query>",
        deps=PydanticAIDeps(event_deps=deps),
        output_type=Output,
    )
    res = await prod_run(deps, run)
    res = res.output
    return list(res.subqueries)


@instrument
async def recursive_splitting_question(
    deps: TaskEventDeps, model: ModelWithSettings, query: str
):
    res = await splitting_question(deps, model, query)
    if len(res) == 1:
        return res
    results: list[list[str] | Exception] = await gather(
        *[
            recursive_splitting_question(deps.spawn(), model, subquery)
            for subquery in res
        ],
    )
    return [
        item
        for sublist in results
        if not isinstance(sublist, Exception)
        for item in sublist
    ]


if __name__ == "__main__":
    import logfire
    from stone_brick.asynclib import gather

    from simplicity.resources import Resource
    from simplicity.utils import get_settings_from_project_root

    logfire.configure()
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx()

    settings = get_settings_from_project_root()
    resource = Resource(settings)
    model = resource.get_llm("google/gemini-2.5-flash")

    tasks = [
        # Multi-faceted queries with temporal elements
        (
            "英国脱欧如何影响了英国的移民政策、与欧盟国家的贸易协定、已经居住在英国的欧盟公民的地位，以及2025-2030年的预期影响是什么",
            (3, 6),
        ),
        # Technical queries with multiple dependencies
        (
            "解释如何在硅谷设立科技创业公司，包括公司注册、外国创始人的签证要求、税务影响、融资策略，以及与在特拉华州注册的比较",
            (4, 7),
        ),
        # Cultural queries mixing unrelated topics
        (
            "京都和东京的日本茶道传统有什么区别，它们如何与现代日本商务礼仪相关，以及西方文化的影响如何改变了这些传统",
            (2, 4),
        ),
        # Nested technical queries
        (
            "创建一个全面的Python教程，涵盖基础语法、高级装饰器、异步编程、元类，以及它们与JavaScript、Ruby和Go中类似功能的比较",
            (4, 8),
        ),
        # Complex financial/legal query
        (
            "解释瑞士银行对非居民的监管规定，包括开户程序、CRS下的税务报告要求、与新加坡和卢森堡的比较，以及由于国际压力而产生的最新变化",
            (4, 6),
        ),
        # Mixed query with unclear boundaries
        (
            "找到曼谷的米其林星级餐厅、他们的招牌菜、价格范围、着装要求、如何预订，并解释米其林指南在亚洲的历史",
            (2, 2),
        ),
        # Philosophical and practical mix
        (
            "人工智能的含义是什么，它与人类智能有何不同，在医疗保健中的实际应用，伦理问题，以及实现简单AI模型的分步指南",
            (4, 6),
        ),
        # Extremely vague query
        ("告诉我关于欧洲的一切", (1, 10)),
        # Query with conditional logic
        (
            "如果我是想在日本工作的美国公民，签证要求是什么，但如果我已经与日本公民结婚，这会如何改变，在那里创业又如何",
            (2, 4),
        ),
        # Meta-query about queries
        (
            "你如何确定何时应该将问题拆分为子查询，查询分解的最佳实践是什么，你能否分析这个问题本身作为例子",
            (2, 4),
        ),
        (
            "如何用Python打印Hello World",
            (1, 1),
        ),
        # Simple but with many details
        (
            "请推荐一本适合初学者的机器学习书籍",
            (1, 1),
        ),
        # Nested but coherent
        (
            "如何准备技术面试（包括算法、系统设计和行为问题）",
            (1, 4),
        ),
        # Looks like it needs splitting but doesn't
        (
            "为什么Python在数据科学领域如此流行",
            (1, 1),
        ),
        # Complex with dependencies that shouldn't be split
        (
            "如何从零开始学习深度学习并在六个月内找到相关工作",
            (1, 3),
        ),
        # Multiple questions that are actually one
        (
            "什么是区块链？它是如何工作的？有哪些应用场景？",
            (1, 3),
        ),
        (
            "比较新加坡、香港和东京的经济政策，同时分析它们的公共交通系统和住房负担能力，以及这些因素对人才吸引力的影响",
            (3, 9),
        ),
        (
            "机器学习中的过拟合问题是什么，如何识别它，以及有哪些常见的解决方法",
            (1, 1),
        ),
    ]

    # async def main():
    #     # Run all in parallel
    #     results = await gather(
    #         *[splitting_question(TaskEventDeps(), model, query) for query, _ in tasks],
    #     )

    #     failed = []

    #     for (query, expected_count), output in zip(tasks, results, strict=True):
    #         if isinstance(output, Exception):
    #             failed.append((query, "Exception", expected_count, "Exception"))
    #             continue
    #         is_failed = (
    #             len(output) < expected_count[0] or len(output) > expected_count[1]
    #         )
    #         if is_failed:
    #             failed.append((query, "Count mismatch", expected_count, len(output)))
    #         print("✓" if not is_failed else "✗", end="")
    #         print(f" Query: {query}")
    #         print(f"  Expected: {expected_count}")
    #         print(f"  Actual: {len(output)}")
    #         print(f"  Subqueries: {output}")

    #     if not failed:
    #         print("All tests passed")
    #     else:
    #         print(f"Failed {len(failed)} out of {len(tasks)} tests")

    # run(main)
