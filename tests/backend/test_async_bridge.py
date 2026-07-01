import asyncio

from symai.backend.async_bridge import run_async


async def _double(x):
    await asyncio.sleep(0)
    return x * 2


def test_run_async_returns_result():
    assert run_async(_double(21)) == 42


def test_run_async_works_inside_running_loop():
    async def _outer():
        return run_async(_double(10))

    assert asyncio.run(_outer()) == 20
