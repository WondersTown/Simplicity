import math
from dataclasses import dataclass, field
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
from copy import copy
from uuid import uuid4

import anyio
from anyio import create_memory_object_stream
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import BaseModel, Field
from typing_extensions import Self

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


class EventTaskStart(Event, Generic[T]):
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
    EventTaskStart | EventTaskOutput | EventTaskOutputStream | EventTaskOutputStreamDelta
)
EventCallbackFunc: TypeAlias = Callable[[Event], Awaitable[None]]


def print_event(event: TaskEvent):
    if isinstance(event, EventTaskStart):
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

TE = TypeVar("TE", bound=TaskEvent)

class DefaultEventCollector(Generic[TE]):
    event_send_stream: MemoryObjectSendStream[TE]
    event_receive_stream: MemoryObjectReceiveStream[TE]

    def __init__(self):
        self.event_send_stream, self.event_receive_stream = create_memory_object_stream(
            max_buffer_size=math.inf
        )

    async def read_event(self):
        async for event in self.event_receive_stream:
            yield event # type: ignore

    async def send_event(self, event: TE):
        await self.event_send_stream.send(event)


@dataclass
class EndResult(Generic[T]):
    res: T

@dataclass
class EventDeps:
    _event_parent: Context = field(default_factory=lambda: Context())
    _event_collector: DefaultEventCollector = field(
        default_factory=DefaultEventCollector
    )
    _event_being_consuming: bool = False

    async def event_send(self, event: Event):
        event.ctx = event.ctx or self._event_parent.spawn()
        await self._event_collector.send_event(event)


    def spawn(self) -> Self:
        another_deps = copy(self)
        another_deps._event_parent = self._event_parent.spawn()
        return another_deps

    async def run(
        self,
        target: Callable[[], Awaitable[T]],
    ) -> AsyncGenerator[tuple[TaskEvent, bool] | EndResult[T], Any]:
        """
        Run the task which produces `TaskEvent` stream,
        and consume the stream to yield events.
        The events are yielded in the following format:
        - `(event, True)` if the event is a part of the result
        - `(event, False)` if the event is not part of the result
        - `EndResult[T]` if the task is finished
        """
        stream_span: None | Context = None
        result: T  = None # type: ignore
        
        async def run_task():
            nonlocal result
            result = await target()
        
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(run_task)
                async for event in self._event_collector.read_event():
                    if stream_span is not None and isinstance(event, EventTaskOutputStreamDelta) and event.ctx.parent_id == stream_span.span_id: # type: ignore
                        yield event, True
                        if event.stopped:
                            break
                        continue
    
                    if event.ctx.parent_id != self._event_parent.span_id: # type: ignore
                        yield event, False
                        continue
    
                    if stream_span is None:
                        if isinstance(event, EventTaskOutputStream) and event.is_result:
                            stream_span = event.ctx
                            yield event, True
                            continue
                        elif (isinstance(event, EventTaskOutput) and event.is_result):
                            yield event, True
                            break
                    yield event, False
            
            yield EndResult[T](res=result)
                
        except* BaseException as exc_group:
            for exc in exc_group.exceptions:
                raise exc from None
    