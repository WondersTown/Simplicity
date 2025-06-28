import re
from dataclasses import dataclass
from datetime import datetime
from logging import getLogger
from typing import Annotated, Literal

from httpx import AsyncClient, HTTPStatusError
from pydantic import BaseModel, Field, field_validator
from stone_brick.asynclib import gather
from stone_brick.observability import instrument
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random
from uuid import uuid4, UUID

logger = getLogger(__name__)


class JinaUsage(BaseModel):
    tokens: int


class JinaMeta(BaseModel):
    usage: JinaUsage


class SearchData(BaseModel):
    id_: str = Field(default_factory=lambda: uuid4().hex)
    title: str
    url: str
    description: str
    usage: JinaUsage


class SearchResponse(BaseModel):
    code: Literal[200]
    data: list[SearchData]
    meta: JinaMeta


# Pydantic models for the Jina Reader API response
class ReaderData(BaseModel):
    id_: str = Field(default_factory=lambda: uuid4().hex)
    title: str
    url: str
    description: str
    content: str
    images: Annotated[dict[str, str], Field(default_factory=dict)]
    links: Annotated[dict[str, str], Field(default_factory=dict)]
    publishedTime: datetime | None = None
    usage: JinaUsage

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


class ReaderResponse(BaseModel):
    code: Literal[200]
    data: ReaderData
    meta: JinaMeta


def clean_md_links(content: str) -> str:
    """
    Clean markdown content by replacing URLs in image and link tags with empty strings.
    Handles nested cases like images with links in alt text.

    Args:
        content: The markdown content to clean

    Returns:
        Cleaned markdown content with URLs removed from image and link tags
    """
    # Simple approach: Replace any ](url) with ]()
    # Works perfectly for markdown, rare false positives in edge cases
    content = re.sub(r"\]\([^)]*\)", r"]()", content)

    return content


@dataclass
class JinaClient:
    api_key: str
    client: AsyncClient

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random(),
        retry=retry_if_exception_type(HTTPStatusError),
    )
    async def search(self, query: str, page: int = 1) -> SearchResponse:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-Respond-With": "no-content",
        }

        response = await self.client.get(
            url="https://s.jina.ai/", params={"q": query, "page": page}, headers=headers
        )
        response.raise_for_status()
        return SearchResponse.model_validate(response.json())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random(),
        retry=retry_if_exception_type(HTTPStatusError),
    )
    async def read(self, target_url: str, timeout: int = 15) -> ReaderResponse:
        """
        Reads the content of a given URL using the Jina Reader API.

        Args:
            target_url: The URL to read content from
            timeout: Timeout in seconds for the request

        Returns:
            ReaderResponse containing the parsed content
        """
        # The Jina Reader API expects the target URL to be appended to its base URL
        # Example: https://r.jina.ai/https://example.com
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-Engine": "cf-browser-rendering",
            "X-With-Generated-Alt": "true",
            "X-With-Images-Summary": "true",
            "X-With-Links-Summary": "true",
            "X-Timeout": str(timeout),
        }

        response = await self.client.get(
            url=f"https://r.jina.ai/{target_url}", headers=headers, timeout=timeout + 10
        )

        response.raise_for_status()

        res = ReaderResponse.model_validate(response.json())
        res.data.content = clean_md_links(res.data.content)
        return res

    @instrument
    async def search_with_read(
        self, query: str, num: int = 9, timeout: int = 15, reader_concurrency: int = 3
    ) -> list[ReaderData]:
        try:
            search_l: list[SearchData] = []
            page = 1
            while len(search_l) < num:
                search_l.extend((await self.search(query, page)).data)
                page += 1
            search_l = search_l[:num]
        except Exception as e:
            raise RuntimeError(f"Failed to search using Jina: {e}") from e

        read_l = await gather(
            *[self.read(result.url, timeout) for result in search_l],
            batch_size=reader_concurrency,
        )
        result_l: list[ReaderData] = []
        for searched, read in zip(search_l, read_l, strict=True):
            if isinstance(read, Exception):
                try:
                    raise read
                except:  # noqa: E722
                    logger.warning(
                        "Failed to read content using Jina from %s: %s",
                        searched.url,
                        read,
                        exc_info=True,
                    )
            else:
                read.data.id_ = searched.id_
                result_l.append(read.data)

        return result_l
