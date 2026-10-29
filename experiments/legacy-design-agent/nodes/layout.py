"""
레이아웃 관련 노드 모듈
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from nodes.common import State
from utils import load_system_template
from dotenv import load_dotenv

load_dotenv()

# -------------------------------  
# 1. Pydantic 모델 정의  
# -------------------------------  

class Layout(BaseModel):  
    """레이아웃 전체 구성을 정의하는 모델"""  
    style_description: str = Field(description="전체 레이아웃 스타일에 대한 설명")  
    elements: Optional[Dict[str, List[int]]] = Field(  
        default_factory=dict,  
        description="레이아웃 요소들의 위치와 크기 (키: 요소 이름, 값: [x, y, width, height])"  
    )

def layout_node(state: State) -> State:  
    """레이아웃을 생성하는 노드 함수"""  
    # design_spec에서 layout_concept 가져오기  
    design_spec = state.get("design_spec", {})  
    layout_concept = design_spec.get("layout_concept", "simple")  
    canvas_size = state.get("canvas_size", (800, 600))
      
    # 시스템 템플릿 로드  
    system_template = load_system_template("prompts/layout_prompt.yaml")  
      
    # 프롬프트 준비 - layout_concept과 canvas_size 추가  
    prompt = ChatPromptTemplate.from_messages([  
        ('system', system_template),  
        ('system', f"""
다음 레이아웃 컨셉을 참고하여 구체적인 레이아웃을 생성해주세요: {layout_concept}
캔버스 크기는 {canvas_size[0]}x{canvas_size[1]}입니다.
        """),  
        MessagesPlaceholder(variable_name='messages'),  
    ])  
      
    # LLM 초기화 및 구조화된 출력 설정  
    llm = ChatOpenAI(model='gpt-4o-mini')
    structured_llm = llm.with_structured_output(Layout)
      
    # 체인 구성: prompt -> structured_llm  
    chain = prompt | structured_llm
      
    # 체인 호출  
    try:
        layout_result = chain.invoke({"messages": state["messages"]})  # Layout 객체 직접 반환
    except Exception as e:
        raise RuntimeError(f"레이아웃 생성 중 오류 발생: {e}")
      
    # LangGraph 패턴에 따라 결과 반환  
    # 1. 레이아웃 결과를 별도의 상태 키에 저장 (Pydantic 객체를 딕셔너리로 변환)
    # 2. 메시지에는 요약 정보 추가  
    return {  
        "layout_spec": layout_result.model_dump()["elements"],  # 레이아웃 결과를 별도 키에 저장  
        "messages": [AIMessage(content=f"레이아웃이 생성되었습니다: {layout_result.style_description}")]  
    }

if __name__ == "__main__":
    # 테스트용 코드
    test_state = {
        "canvas_size": (1200, 800),  
        "design_spec": {
            "layout_concept": "대칭적 구조로 중심에 참이슬 로고를 배치하고, 주변에 자연 요소를 배치하여 전통과 현대의 조화를 이루는 심플한 디자인"
        },
        "messages": [("user", "디자인 컨셉 '신선한 자연과 전통의 조화'이(가) 생성되었습니다.")]
    }
    result = layout_node(test_state)
    print("layout_spec:", result["layout_spec"])
    print("messages:", result["messages"][0].content)