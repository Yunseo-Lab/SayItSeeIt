# layout_generator.py
import asyncio
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

async def _generate_one(prompt: str):
    # 실제 HTTP 요청을 백그라운드 스레드에서 실행
    return (await asyncio.to_thread(
        client.responses.create,
        model="gpt-4.1-mini",
        input=[{
            "role": "system",
            "content": [{"type": "input_text", "text": prompt}]
        }],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[],
        temperature=1,
        max_output_tokens=2048,
        top_p=1,
        store=False
    )).output_text

async def _generate_many(prompt: str, n: int):
    tasks = [_generate_one(prompt) for _ in range(n)]
    return await asyncio.gather(*tasks)

def generate_layouts(prompt: str, n: int = 5) -> list[str]:
    """
    동기 함수로 호출 가능.
    내부적으로 asyncio 이벤트 루프를 띄워서 _generate_many() 를 실행합니다.
    """
    return asyncio.run(_generate_many(prompt, n))


if __name__ == "__main__":

    query = "Create a layout for a modern web application dashboard with a sidebar, header, and main content area."
    result = generate_layouts(query, 3)

    print("Generated Layouts:")
    for i, layout in enumerate(result, 1):
        print(f"Layout {i}:\n{layout}\n")
