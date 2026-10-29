"""
그리기 관련 노드 모듈
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from nodes.common import State, drawer_tools
from utils import load_system_template


def drawer(state: State) -> State:
    """그림을 생성하는 노드 함수"""
    system_template = load_system_template("prompts/drawer_prompt.yaml")

    canvas_size = state.get("canvas_size", (800, 600))
    design_spec = state.get("design_spec", {})
    layout_spec = state.get("layout_spec", {})
    background_spec = state.get("background_spec", "simple")

    # 디자인 스펙의 각 키에 대해 안전하게 접근
    concept = design_spec.get("concept", "")
    description = design_spec.get("description", "")
    colors = design_spec.get("colors", [])
    fonts = design_spec.get("fonts", [])
    background_concept = design_spec.get("background_concept", "")
    layout_concept = design_spec.get("layout_concept", "")
    decoration_icons = design_spec.get("decoration_icons", [])

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        (
            "user",
            # 여기에서 f-string 대신 format_template() 사용
            # f-string 내부에 중괄호가 있을 경우 템플릿이 중괄호를 변수로 잘못 해석할 수 있음
            """
            캔버스 크기: {canvas_size}
            컨셉: {concept}
            설명: {description}
            컬러: {colors}
            폰트: {fonts}
            배경 컨셉: {background_concept}
            레이아웃 컨셉: {layout_concept}
            장식 아이콘: {decoration_icons}
            레이아웃: {layout_spec}
            배경 스펙: {background_spec}
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]).partial(
        canvas_size=canvas_size,
        concept=concept,
        description=description,
        colors=colors,
        fonts=fonts,
        background_concept=background_concept,
        layout_concept=layout_concept,
        decoration_icons=decoration_icons,
        layout_spec=layout_spec,
        background_spec=background_spec
    )
    llm = ChatOpenAI(model="gpt-4o-mini")
    # llm_with_tools = llm.bind_tools(drawer_tools)
    chain = prompt | llm
    
    # 메시지 안전하게 접근
    messages = state.get("messages", [])
    result = chain.invoke({"messages": messages})
    
    return {"messages": [result]}