"""
배경 관련 노드 모듈
"""
from langchain_core.messages import AIMessage
from nodes.common import State

def background_node(state: State) -> State:
    """배경을 생성하는 노드 함수"""
    design_spec = state.get("design_spec", {})
    background_concept = design_spec.get("background_concept", "simple")
    
    # tmp
    color = design_spec.get("colors", "#FFFFFF")  # 기본 색상
    background = color[0]
    # tmp

    return {
        "background_spec": background,  # 배경 결과를 별도 키로 반환
        "messages": [AIMessage(content=f"배경 '{background_concept}'이(가) 생성되었습니다.")]
    }
