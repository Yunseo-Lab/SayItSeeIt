from layoutgenerator import TextToLayoutPipeline
from src.generators.copy_generator import generate_copies

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import invoke_graph, stream_graph, random_uuid
from langchain_teddynote.graphs import visualize_graph



# GraphState 상태 정의
class GraphState(TypedDict):
    pipeline: Annotated[TextToLayoutPipeline, "Pipeline"]  # 파이프라인 인스턴스
    query: Annotated[str, "Query"]  # 사용자 요청 사항
    layout: Annotated[list, "Layout"]  # 레이아웃 생성 결과
    copy: Annotated[list, "Copy"]  # 카피 생성 결과
    # messages: Annotated[list, add_messages]  # 메시지(누적되는 list)


# 레이아웃 생성 노드
def generate_layouts_node(state: GraphState) -> GraphState:
    pipeline = state["pipeline"]
    user_text = state["query"]
    
    results = pipeline.run(user_text=user_text)

    return {**state, "layout": results}


# 카피 생성 노드
def generate_copies_node(state: GraphState) -> GraphState:
    pipeline = state["pipeline"]
    user_text = state["query"]
    layout = state["layout"]

    layout_lists = [list(sample.keys()) for sample in pipeline.map_labels_to_bboxes(layout)]
    copy_result = generate_copies(layout_lists, user_text)

    return {**state, "copy": copy_result}


# 레이아웃 시각화 노드
def visualize_layouts_node(state: GraphState) -> GraphState:
    pipeline = state["pipeline"]
    layout = state["layout"]
    copy_result = state["copy"]

    # 시각화
    pipeline.visualize(layout, copy=copy_result, show_bbox=True)

    return state


from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

def initialize_graph():

    workflow = StateGraph(GraphState)

    workflow.add_node("layout_generation", generate_layouts_node)
    workflow.add_node("copy_generation", generate_copies_node)
    workflow.add_node("visualization", visualize_layouts_node)

    workflow.add_edge("layout_generation", "copy_generation")
    workflow.add_edge("copy_generation", "visualization")
    workflow.add_edge("visualization", END)

    # 그래프 진입점 설정
    workflow.set_entry_point("layout_generation")

    # 컴파일
    app = workflow.compile()

    visualize_graph(app)

    return app

if __name__ == "__main__":
    # 파이프라인 초기화
    pipeline = TextToLayoutPipeline(dataset="cardnews")

    # 초기 상태 정의
    initial_state: GraphState = {
        "pipeline": pipeline,
        "query": "빙그레 초코우유를 홍보하는 카드뉴스를 만들어줘.",
        "layout": [],
        "copy": [],
    }


    # config 설정(재귀 최대 횟수, thread_id)
    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})

    app = initialize_graph()

    # 그래프 실행
    invoke_graph(app, initial_state, config)

    # 그래프를 스트리밍 출력
    stream_graph(app, initial_state, config)
