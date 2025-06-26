from contextlib import AbstractAsyncContextManager
from copy import copy
from dataclasses import dataclass, field
from functools import wraps
from inspect import isawaitable
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    TypeVar,
)

from pydantic_ai import RunContext
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.result import StreamedRunResult
from typing_extensions import Concatenate, ParamSpec

from simplicity.common.event import (
    Context,
    DefaultEventCollector,
    Event,
    EventTaskOutput,
    EventTaskOutputStream,
    EventTaskOutputStreamDelta,
    EventTaskStart,
    TaskEvent,
    EndResult,
    EventDeps,
)

@dataclass
class PydanticAIDeps:
    event_deps: EventDeps = field(default_factory=EventDeps)


T = TypeVar("T")


def fork_pydantic_ai_ctx(ctx: RunContext[PydanticAIDeps]) -> RunContext[PydanticAIDeps]:
    new_ctx = copy(ctx)
    new_ctx.deps.event_deps = ctx.deps.event_deps.spawn()
    return new_ctx


async def prod_run_stream(
    event_deps: EventDeps, stream: AbstractAsyncContextManager[StreamedRunResult[PydanticAIDeps, T]]
) -> StreamedRunResult[PydanticAIDeps, T]:
    """Run the agent and produce `TaskEvent` stream into a channel inside the `deps`. Finally return a `StreamedRunResult`."""
    await event_deps.event_send(
        EventTaskStart(
            ctx=event_deps._event_parent, task_desc="run_agent_stream", task_args={}
        )
    )

    stream_span = event_deps._event_parent.spawn()
    await event_deps.event_send(EventTaskOutputStream(ctx=stream_span, is_result=True))

    async with stream as response:
        # Do streaming
        async for result in response.stream_text(delta=True):
            await event_deps.event_send(
                EventTaskOutputStreamDelta(
                    ctx=stream_span.spawn(), task_output_delta=result, stopped=False
                )
            )
        await event_deps.event_send(
            EventTaskOutputStreamDelta(
                ctx=stream_span.spawn(), task_output_delta="", stopped=True
            )
        )
        return response


async def prod_run(
    deps: EventDeps, run: Awaitable[AgentRunResult[T]]
) -> AgentRunResult[T]:
    """Run the agent and produce `TaskEvent` stream into a channel inside the `deps`. Finally return a `AgentRunResult`."""
    await deps.event_send(
        EventTaskStart(ctx=deps._event_parent, task_desc="run_agent", task_args={})
    )
    # Run
    result = await run
    await deps.event_send(EventTaskOutput(task_output=result.output, is_result=True))
    return result


P = ParamSpec("P")


def with_events(
    func: Callable[Concatenate[RunContext[PydanticAIDeps], P], Awaitable[T]],
) -> Callable[Concatenate[RunContext[PydanticAIDeps], P], Awaitable[T]]:
    """Decorator that wraps a function with event handling logic"""

    @wraps(func)
    async def wrapper(ctx: RunContext[PydanticAIDeps], *args: P.args, **kwargs: P.kwargs):
        # Spawn root span
        new_ctx = fork_pydantic_ai_ctx(ctx)
        await new_ctx.deps.event_deps.event_send(
            EventTaskStart(
                ctx=new_ctx.deps.event_deps._event_parent,
                task_desc=func.__name__,
                task_args={"args": args, "kwargs": kwargs},
            )
        )

        # Execute the actual function
        func_call = func(new_ctx, *args, **kwargs)
        if isawaitable(func_call):
            result = await func_call
        else:
            result = func_call

        # Create and send output event
        await new_ctx.deps.event_deps.event_send(
            EventTaskOutput(task_output=result, is_result=True)
        )
        return result

    return wrapper


async def agent_run(
    deps: EventDeps,
    run: Awaitable[AgentRunResult[T]],
) -> AsyncGenerator[tuple[TaskEvent, bool] | EndResult[AgentRunResult[T]], Any]:
    """Start running the agent and yield events.
    It internally uses `prod_run` to run the agent which produces `TaskEvent` stream,
    and consuming the stream to yield events.
    """
    async for event in deps.consume(
        lambda: prod_run(deps, run)
    ):
        yield event


async def agent_run_stream(
    event_deps: EventDeps,
    stream: AbstractAsyncContextManager[StreamedRunResult[PydanticAIDeps, T]],
) -> AsyncGenerator[tuple[TaskEvent, bool] | EndResult[StreamedRunResult[PydanticAIDeps, T]], Any]:
    """Start running the agent and yield events.
    It internally uses `prod_run_stream` to run the agent which produces `TaskEvent` stream,
    and consuming the stream to yield events.
    """
    async for event in event_deps.consume(
        lambda: prod_run_stream(event_deps, stream)
    ):
        yield event  
