# generators/copy_generator.py

import asyncio
import json
from typing import List, Dict
from .openai_client import get_client

async def _gen_one_copy(layout_elems: List[str], user_text: str) -> Dict[str, str]:
    client = get_client()

    schema = {
        "type": "object",
        "properties": {
            elem: {
                "type": "string",
                "description": f"레이아웃 요소 `{elem}` 에 어울리는 한글 홍보 문구"
            }
            for elem in layout_elems
        },
        "required": layout_elems,
        "additionalProperties": False
    }

    prompt = (
        "다음 레이아웃 요소에 대응하는 한글 홍보 문구를 각각 생성해주세요:\n"
        f"{layout_elems}\n\n"
        "반드시 JSON 객체 형식으로, 키는 요소 이름, 값은 해당 문구여야 합니다."
    )

    resp = await asyncio.to_thread(
        client.responses.create,
        model="gpt-4o-mini",
        input=[{"role": "system", "content": [{"type": "input_text", "text": prompt}]},
               {"role": "user", "content": [{"type": "input_text", "text": user_text}]}],
        text={
            "format": {
                "name": "copy_schema",
                "type": "json_schema",
                "strict": True,
                "schema": schema,
            }
        },
        reasoning={},
        tools=[],
        temperature=0.8,
        max_output_tokens=512,
        top_p=1,
        store=False,
    )

    return json.loads(resp.output_text)


async def _gen_copies_async(layouts: List[List[str]], user_text: str) -> List[Dict[str, str]]:
    tasks = [_gen_one_copy(elems, user_text) for elems in layouts]
    return await asyncio.gather(*tasks)


def generate_copies(layouts: List[List[str]], user_text: str) -> List[Dict[str, str]]:
    """
    :param layouts: [['title','description','text','icon'], ...]
    :param user_text: 사용자 요구사항 텍스트
    :return: [{'title': "...", 'description': "...", ...}, ...]
    """
    return asyncio.run(_gen_copies_async(layouts, user_text))


if __name__ == "__main__":
    # 예시 레이아웃 정의
    example_layouts = [
        ["title", "description", "text", "icon"],
        ["header", "button", "footer"]
    ]
    # 예시 사용자 요구사항
    example_user_text = "친환경 제품을 홍보하는 웹사이트를 만들고 싶습니다. 자연친화적이고 신뢰감 있는 느낌으로 부탁드립니다."
    
    # 문구 생성
    try:
        copies = generate_copies(example_layouts, example_user_text)
        for i, copy in enumerate(copies, 1):
            print(f"--- Copy {i} ---")
            for key, val in copy.items():
                print(f"{key}: {val}")
            print()
    except Exception as e:
        print("문구 생성 중 에러 발생:", e)
