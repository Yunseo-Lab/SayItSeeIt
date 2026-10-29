from typing import List, Dict, Annotated, Optional  
from pydantic import BaseModel, Field  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from langchain_openai import ChatOpenAI  
from langchain_core.messages import AIMessage  
from typing_extensions import TypedDict  
  
from nodes.common import State, designer_tools  
from utils import load_system_template  
from langgraph.graph.message import add_messages  
  
# -------------------------------  
# 1. Pydantic 모델 정의  
# -------------------------------  
class DesignSpec(BaseModel):  
    """  
    디자인 요소를 구조화된 형태로 표현하는 모델  
    """  
    concept: str = Field(description="디자인의 핵심 컨셉")  
    description: str = Field(description="컨셉에 대한 상세 설명")  
    colors: List[str] = Field(description="주요 색상 리스트 (HEX 코드 또는 CSS 색상 이름)")  
    fonts: List[str] = Field(description="추천 글꼴 리스트")  
    background_concept: str = Field(description="배경에 대한 컨셉 설명")  
    layout_concept: str = Field(description="레이아웃 구성 방식 설명")  
    decoration_icons: List[str] = Field(description="추가 장식 아이콘 제안 리스트")  
  
# -------------------------------  
# 2. 상태 정의 (LangGraph 패턴에 맞게)  
# -------------------------------  
# class DesignerState(TypedDict):  
#     messages: Annotated[List, add_messages]  # 대화 흐름을 위한 메시지  
#     design_spec: Optional[Dict]  # 디자인 스펙 결과를 저장하는 별도 키  
#     canvas_size: tuple  # 캔버스 크기 정보  
  
# -------------------------------  
# 3. 디자이너 노드 함수 구현  
# -------------------------------  
def designer(state: State) -> State:  
    """  
    주어진 대화 상태(`state['messages']`)를 기반으로 디자인 스펙을 생성합니다.  
    LLM에게 구조화된 출력을 요청하고 Pydantic 모델로 파싱하여 반환합니다.  
    """  
    # 시스템 프롬프트 로드  
    system_template: str = load_system_template(
        "prompts/designer_prompt.yaml"
    )
  
    # 프롬프트 구성  
    prompt = ChatPromptTemplate.from_messages([  
        ("system", system_template),  
        ("system", "디자인 스펙을 구조화된 형식으로 생성해주세요."),  
        MessagesPlaceholder(variable_name="messages"),  
    ])  
  
    # LLM 세팅 - 최신 방식으로 structured_output 사용  
    llm = ChatOpenAI(model="gpt-4o-mini")  
      
    # designer_tools가 필요한 경우 도구 바인딩  
    if designer_tools:  
        llm_with_tools = llm.bind_tools(designer_tools)  
        # with_structured_output 메서드로 구조화된 출력 요청  
        structured_llm = llm_with_tools.with_structured_output(DesignSpec)  
    else:  
        # 도구가 필요 없는 경우 직접 structured_output 사용  
        structured_llm = llm.with_structured_output(DesignSpec)  
      
    # 체인 구성: prompt -> structured_llm  
    chain = prompt | structured_llm  
      
    # 체인 실행 및 결과 파싱  
    try:  
        design_spec = chain.invoke({"messages": state["messages"]})  # DesignSpec 객체 직접 반환  
    except Exception as e:  
        raise RuntimeError(f"디자인 생성 중 오류 발생: {e}")  
  
    # print("디자인 스펙:", design_spec)  
  
    # 결과를 별도의 상태 키에 저장하고, 메시지에는 요약 정보만 추가  
    return {  
        "design_spec": design_spec.model_dump(),  # 전체 디자인 스펙을 별도 키에 저장 (Pydantic v2 방식)  
        "messages": [AIMessage(content=f"디자인 컨셉 '{design_spec.concept}'이(가) 생성되었습니다.")]  
    }  
  
  
if __name__ == "__main__":  
    example_state = {  
        "canvas_size": (800, 600),  
        "messages": [("user", "참이슬 패키지 디자인 컨셉을 제안해주세요.")]  
    }  
    result = designer(example_state)  
    print("디자인 스펙 결과:", result["design_spec"])  
    print("메시지:", result["messages"][0].content)

# -------------------------------
'''디자인 스펙: 
concept='신선한 자연과 전통의 조화' 
description='참이슬의 패키지 디자인은 한국의 청정 자연에서 영감을 받아, 전통과 현대를 아우르는 느낌을 강조합니다. 깨끗하고 투명한 물을 상징하며, 소비자에게 품질과 신뢰를 전달할 것입니다.' 
colors=['#A3D6E9', '#005E75', '#FFFFFF'] 
fonts=['Noto Sans KR Regular', 'Noto Serif KR Bold'] 
background_concept='자연에서 영감을 받은 청량감 있는 배경, 맑은 물과 푸른 하늘을 나타내는 색감으로 소비자에게 시원함을 전달' 
layout_concept='대칭적 구조로 중심에 참이슬 로고를 배치하고, 주변에 자연 요소를 배치하여 전통과 현대의 조화를 이루는 심플한 디자인' 
decoration_icons=['물방울 아이콘', '자연 요소 아이콘', '한국 전통 문양 아이콘']


# Returned state:
canvas_size: (800, 600)
design_spec: {'concept': '신선한 자연과 전통의 조화', 'description': '참이슬의 패키지 디자인은 한국의 청정 자연에서 영감을 받아, 전통과 현대를 아우르는 느낌을 강조합니다. 깨끗하고 투명한 물을 상징하며, 소비자에게 품질과 신뢰를 전달할 것입니다.', 'colors': ['#A3D6E9', '#005E75', '#FFFFFF'], 'fonts': ['Noto Sans KR Regular', 'Noto Serif KR Bold'], 'background_concept': '자연에서 영감을 받은 청량감 있는 배경, 맑은 물과 푸른 하늘을 나타내는 색감으로 소비자에게 시원함을 전달', 'layout_concept': '대칭적 구조로 중심에 참이슬 로고를 배치하고, 주변에 자연 요소를 배치하여 전통과 현대의 조화를 이루는 심플한 디자인', 'decoration_icons': ['물방울 아이콘', '자연 요소 아이콘', '한국 전통 문양 아이콘']}
messages: 디자인 컨셉 '신선한 자연과 전통의 조화'이(가) 생성되었습니다.'''