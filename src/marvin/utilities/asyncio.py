import asyncio
from typing import Any, Coroutine, TypeVar

from marvin.settings import settings

if settings.apply_nest_asyncio:
    import nest_asyncio

    nest_asyncio.apply()  # type: ignore

T = TypeVar("T")


def run_sync(coroutine: Coroutine[Any, Any, T]) -> T:
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coroutine)
        return loop.run_until_complete(task)
    except RuntimeError:  # No running loop
        return asyncio.run(coroutine)
