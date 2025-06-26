import math
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Generic,
    Literal,
    Optional,
    TypeAlias,
    TypeVar,
)
from uuid import uuid4

import anyio
from anyio import create_memory_object_stream
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import BaseModel, Field

T = TypeVar("T")


class Context(BaseModel):
    trace_id: int = Field(default_factory=lambda: uuid4().int)
    span_id: int = Field(default_factory=lambda: uuid4().int)
    parent_id: Optional[int] = None

    def spawn(self) -> "Context":
        return Context(
            trace_id=self.trace_id,
            span_id=uuid4().int,
            parent_id=self.span_id,
        )


class Event(BaseModel):
    ctx: Context | None = None
    event_type: Literal[
        "task_start", "task_output", "task_output_stream", "task_output_delta"
    ]


# Events should be like:
# |
# |-- EventTaskRoot
# |        |
# |        |-- EventTaskOutput
# |        |-- EventTaskOutput
# |       ...
# |
# |-- EventTaskRoot
#          |-- EventTaskOutputStream (optional)
#          |            |
#          |            |-- EventTaskOutputStreamDelta
#          |            |
#          |            |-- EventTaskOutputStreamDelta
#          |            |
#          |           ...
#         ...


class EventTaskRoot(Event, Generic[T]):
    """Represents a task root event"""

    event_type: Literal["task_start"] = "task_start"
    task_desc: str
    task_args: T | Any


class EventTaskOutput(Event, Generic[T]):
    """Represents a task output event"""

    event_type: Literal["task_output"] = "task_output"
    is_result: bool = False
    task_output: T | Any


class EventTaskOutputStream(Event):
    """Represents a task output stream event"""

    event_type: Literal["task_output_stream"] = "task_output_stream"
    is_result: bool = False


class EventTaskOutputStreamDelta(Event, Generic[T]):
    """Represents a task output delta event"""

    event_type: Literal["task_output_delta"] = "task_output_delta"
    task_output_delta: T | Any
    stopped: bool = False

    def get_text(self) -> str | None:
        return str(self.task_output_delta)


TaskEvent: TypeAlias = (
    EventTaskRoot | EventTaskOutput | EventTaskOutputStream | EventTaskOutputStreamDelta
)
EventCallbackFunc: TypeAlias = Callable[[Event], Awaitable[None]]


def print_event(event: TaskEvent):
    if isinstance(event, EventTaskRoot):
        print(f"Task call: {event.task_desc} with args: \n{event.task_args}")
    elif isinstance(event, EventTaskOutput):
        print(f"Task output: {event.task_output}")
    elif isinstance(event, EventTaskOutputStream):
        print("Task output stream:")
    elif isinstance(event, EventTaskOutputStreamDelta):
        if event.stopped:
            print()
        else:
            print(event.task_output_delta, end="", flush=True)


class DefaultEventCollector:
    event_send_stream: MemoryObjectSendStream[Event]
    event_receive_stream: MemoryObjectReceiveStream[Event]

    def __init__(self):
        self.event_send_stream, self.event_receive_stream = create_memory_object_stream(
            max_buffer_size=math.inf, item_type=Event
        )

    async def send_event(self, event: Event):
        await self.event_send_stream.send(event)


@dataclass
class EndResult(Generic[T]):
    res: T

async def default_run(
    root_span: Context,
    collector: DefaultEventCollector,
    run: Callable[[], Awaitable[T]],
) -> AsyncGenerator[TaskEvent | EndResult[T], Any]:
    stream_span: None | Context = None
    result: T  = None # type: ignore
    
    async def run_task():
        nonlocal result
        result = await run()
    
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_task)
            async for event in collector.event_receive_stream:
                yield event  # type: ignore
                if (
                    stream_span is None
                    and isinstance(event, EventTaskOutputStream)
                    and event.ctx.parent_id == root_span.span_id  # type: ignore
                ):
                    stream_span = event.ctx
                if (
                    isinstance(event, EventTaskOutput)
                    and event.is_result
                    and event.ctx.parent_id == root_span.span_id  # type: ignore
                ):
                    break
                if (
                    stream_span is not None
                    and isinstance(event, EventTaskOutputStreamDelta)
                    and event.ctx.parent_id == stream_span.span_id  # type: ignore
                    and event.stopped
                ):
                    break
        
        yield EndResult[T](res=result)
            
    except* BaseException as exc_group:
        for exc in exc_group.exceptions:
            raise exc from None
