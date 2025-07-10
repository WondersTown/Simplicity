import re
from asyncio import Semaphore, sleep
from dataclasses import dataclass, field
from logging import getLogger
from typing import Literal

from httpx import AsyncClient, HTTPStatusError
from pydantic import BaseModel
from stone_brick.asynclib import gather
from stone_brick.observability import instrument
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random

from simplicity.structure import ReaderData, SearchData

logger = getLogger(__name__)


class JinaUsage(BaseModel):
    tokens: int


class JinaMeta(BaseModel):
    usage: JinaUsage


class SearchResponse(BaseModel):
    code: Literal[200]
    data: list[SearchData]
    meta: JinaMeta


# Pydantic models for the Jina Reader API response
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

READ_TIMEOUT = 10
# READ_DELAY = 0.333

@dataclass
class JinaClient:
    api_key: str
    client: AsyncClient
    concurrency: int
    read_timeout: int = READ_TIMEOUT
    _semaphore: Semaphore = field(init=False)

    def __post_init__(self):
        self._semaphore = Semaphore(self.concurrency)

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
        retry=retry_if_exception_type(HTTPStatusError),
    )
    async def read(
        self,
        target: str | SearchData,
    ) -> ReaderResponse:
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
            # "X-With-Generated-Alt": "true",
            "X-Retain-Images": "none",
            # "X-With-Images-Summary": "true",
            # "X-With-Links-Summary": "true",
            # "X-Timeout": str(self.read_timeout - 2),
        }

        target_url = target.url if isinstance(target, SearchData) else target
        async with self._semaphore:
            # await sleep((self.concurrency - self._semaphore._value - 1) * READ_DELAY)
            response = await self.client.get(
                url=f"https://r.jina.ai/{target_url}",
                headers=headers,
                timeout=self.read_timeout,
            )

        response.raise_for_status()

        res = ReaderResponse.model_validate(response.json())
        res.data.content = clean_md_links(res.data.content)

        if isinstance(target, SearchData):
            res.data.id_ = target.id_
            res.data.title = target.title
            res.data.description = target.description
            res.data.url = target.url
        return res

    @instrument
    async def read_batch(
        self, targets: list[str | SearchData]
    ) -> list[ReaderResponse | Exception]:
        return await gather(
            *[self.read(target) for target in targets],
        )

    @instrument
    async def search_with_read(
        self, query: str, num: int = 10
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
            *[self.read(result) for result in search_l],
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
