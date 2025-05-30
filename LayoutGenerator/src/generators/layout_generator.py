import asyncio
from typing import List
from .openai_client import get_client

async def _gen_one_layout(prompt: str) -> str:
    client = get_client()
    resp = await asyncio.to_thread(
        client.responses.create,
        model="gpt-4.1-mini",
        input=[{"role":"system","content":[{"type":"input_text","text":prompt}]}],
        text={"format":{"type":"text"}},
        reasoning={},
        tools=[],
        temperature=1,
        max_output_tokens=2048,
        top_p=1,
        store=False,
    )
    return resp.output_text

async def _gen_layouts_async(prompt: str, n: int) -> List[str]:
    tasks = [_gen_one_layout(prompt) for _ in range(n)]
    return await asyncio.gather(*tasks)

def generate_layouts(prompt: str, n: int = 5) -> List[str]:
    return asyncio.run(_gen_layouts_async(prompt, n))
