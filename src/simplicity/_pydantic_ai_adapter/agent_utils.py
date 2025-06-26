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
    EventTaskRoot,
    TaskEvent,
    EndResult,
    default_run,
)


@dataclass
class BaseDeps:
    _event_parent: Context = field(default_factory=lambda: Context())
    _event_collector: DefaultEventCollector = field(
        default_factory=DefaultEventCollector
    )
    _event_being_consuming: bool = False

    async def event_send(self, event: Event):
        event.ctx = event.ctx or self._event_parent.spawn()
        await self._event_collector.send_event(event)


T = TypeVar("T")


def fork_deps(deps: BaseDeps) -> BaseDeps:
    another_deps = copy(deps)
    another_deps._event_parent = deps._event_parent.spawn()
    return another_deps


def fork_pydantic_ai_ctx(ctx: RunContext[BaseDeps]) -> RunContext[BaseDeps]:
    new_ctx = copy(ctx)
    new_ctx.deps = fork_deps(ctx.deps)
    return new_ctx


async def prod_run_stream(
    deps: BaseDeps, stream: AbstractAsyncContextManager[StreamedRunResult[BaseDeps, T]]
) -> StreamedRunResult[BaseDeps, T]:
    """Run the agent and produce `TaskEvent` stream into a channel inside the `deps`. Finally return a `StreamedRunResult`."""
    await deps.event_send(
        EventTaskRoot(
            ctx=deps._event_parent, task_desc="run_agent_stream", task_args={}
        )
    )

    stream_span = deps._event_parent.spawn()
    await deps.event_send(EventTaskOutputStream(ctx=stream_span, is_result=True))

    async with stream as response:
        # Do streaming
        async for result in response.stream_text(delta=True):
            await deps.event_send(
                EventTaskOutputStreamDelta(
                    ctx=stream_span.spawn(), task_output_delta=result, stopped=False
                )
            )
        await deps.event_send(
            EventTaskOutputStreamDelta(
                ctx=stream_span.spawn(), task_output_delta="", stopped=True
            )
        )
        return response


async def prod_run(
    deps: BaseDeps, run: Awaitable[AgentRunResult[T]]
) -> AgentRunResult[T]:
    """Run the agent and produce `TaskEvent` stream into a channel inside the `deps`. Finally return a `AgentRunResult`."""
    await deps.event_send(
        EventTaskRoot(ctx=deps._event_parent, task_desc="run_agent", task_args={})
    )
    # Run
    result = await run
    await deps.event_send(EventTaskOutput(task_output=result.output, is_result=True))
    return result


P = ParamSpec("P")


def with_events(
    func: Callable[Concatenate[RunContext[BaseDeps], P], Awaitable[T]],
) -> Callable[Concatenate[RunContext[BaseDeps], P], Awaitable[T]]:
    """Decorator that wraps a function with event handling logic"""

    @wraps(func)
    async def wrapper(ctx: RunContext[BaseDeps], *args: P.args, **kwargs: P.kwargs):
        # Spawn root span
        new_ctx = fork_pydantic_ai_ctx(ctx)
        await new_ctx.deps.event_send(
            EventTaskRoot(
                ctx=new_ctx.deps._event_parent,
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
        await new_ctx.deps.event_send(
            EventTaskOutput(task_output=result, is_result=True)
        )
        return result

    return wrapper


async def agent_run(
    deps: BaseDeps,
    run: Awaitable[AgentRunResult[T]],
) -> AsyncGenerator[TaskEvent | EndResult[AgentRunResult[T]], Any]:
    """Start running the agent and yield events.
    It internally uses `prod_run` to run the agent which produces `TaskEvent` stream,
    and consuming the stream to yield events.
    """
    if deps._event_being_consuming:
        raise RuntimeError("TaskEvent stream being consuming. Use run_stream instead.")
    deps._event_being_consuming = True
    async for event in default_run(
        deps._event_parent, deps._event_collector, lambda: prod_run(deps, run)
    ):
        yield event 


async def agent_run_stream(
    deps: BaseDeps,
    stream: AbstractAsyncContextManager[StreamedRunResult[BaseDeps, T]],
) -> AsyncGenerator[TaskEvent | EndResult[StreamedRunResult[BaseDeps, T]], Any]:
    """Start running the agent and yield events.
    It internally uses `prod_run_stream` to run the agent which produces `TaskEvent` stream,
    and consuming the stream to yield events.
    """
    if deps._event_being_consuming:
        raise RuntimeError("TaskEvent stream being consuming. Use agent_run instead.")
    deps._event_being_consuming = True
    async for event in default_run(
        deps._event_parent, deps._event_collector, lambda: prod_run_stream(deps, stream)
    ):
        yield event  
