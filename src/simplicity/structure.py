from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, Self, Sequence, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from stone_brick.llm import TaskEvent, TaskEventDeps, TaskOutput


class SearchData(BaseModel):
    kind: Literal["search"] = "search"
    id_: str = Field(default_factory=lambda: str(uuid4())[:6])
    title: str
    url: str
    description: str

    def llm_dump(self) -> dict:
        return self.model_dump()


class ReaderData(SearchData):
    kind: Literal["reader"] = "reader"
    content: str
    images: Annotated[dict[str, str], Field(default_factory=dict)]
    links: Annotated[dict[str, str], Field(default_factory=dict)]
    publishedTime: datetime | None = None

    @field_validator("publishedTime", mode="before")
    @classmethod
    def parse_published_time(cls, v):
        if v is None or isinstance(v, datetime):
            return v

        if isinstance(v, str):
            # Try to parse the format: "2019-07-17 14:31:44 -0400"
            try:
                return datetime.strptime(v, "%Y-%m-%d %H:%M:%S %z")
            except ValueError:
                # If that fails, try without timezone
                try:
                    return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    # If all parsing fails, return None
                    return None

        return v

    def llm_dump(self) -> dict:
        return self.model_dump(exclude={"images", "links"})


class QAData(ReaderData):
    kind: Literal["qa"] = "qa"
    query: str
    answer: str

    def llm_dump(self) -> dict:
        return self.model_dump(
            exclude={
                "images",
                "links",
                # For the raw reader data
                "title",
                "description",
                "content",
            }
        )



@dataclass(slots=True, kw_only=True)
class LLMUsage:
    kind: Literal["llm_usage"] = "llm_usage"
    input_tokens: int | None
    output_tokens: int | None
    config_name: str

InfoData: TypeAlias = ReaderData | SearchData | QAData 

@pydantic_dataclass(slots=True)
class SimpOutput:
    d: Annotated[InfoData| LLMUsage , Field(discriminator="kind")]

    @classmethod
    def gen(cls, d: Sequence[InfoData| LLMUsage]) -> Sequence[Self]:
        return [cls(d=x) for x in d]


SimpTaskOutput: TypeAlias = TaskOutput[Sequence[SimpOutput]]
SimpTaskEvent: TypeAlias = TaskEvent[Sequence[SimpOutput]]
SimpTaskDeps: TypeAlias = TaskEventDeps[Sequence[SimpOutput]]