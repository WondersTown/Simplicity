from datetime import datetime
from typing import Annotated, Sequence, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator
from stone_brick.llm import TaskEvent, TaskEventDeps


class SearchData(BaseModel):
    id_: str = Field(default_factory=lambda: str(uuid4())[:6])
    title: str
    url: str
    description: str

    def llm_dump(self) -> dict:
        return self.model_dump()


class ReaderData(SearchData):
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


InfoData: TypeAlias = ReaderData | SearchData | QAData
OutputDataType: TypeAlias = Sequence[InfoData]

SimplicityTask: TypeAlias = TaskEvent[OutputDataType]
SimplicityTaskOutput: TypeAlias = TaskEventDeps[OutputDataType]
SimplicityTaskDeps: TypeAlias = TaskEventDeps[OutputDataType]