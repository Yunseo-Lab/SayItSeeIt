from layoutgenerator import TextToLayoutPipeline
from src.generators.copy_generator import generate_copies

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import invoke_graph, stream_graph, random_uuid
from src.visualization import Visualizer

# 파이프라인 초기화
dataset = "cardnews"  # 사용할 데이터셋 이름
pipeline = TextToLayoutPipeline(dataset=dataset)


# GraphState 상태 정의
class GraphState(TypedDict):
    pipeline: Annotated[TextToLayoutPipeline, "Pipeline"]  # 파이프라인 인스턴스
    query: Annotated[str, "Query"]  # 사용자 요청 사항
    layout: Annotated[list, "Layout"]  # 레이아웃 생성 결과
    copy: Annotated[list, "Copy"]  # 카피 생성 결과
    images: Annotated[list, "Images"]  # 이미지 목록
    # messages: Annotated[list, add_messages]  # 메시지(누적되는 list)


# 레이아웃 생성 노드
def generate_layouts_node(state: GraphState) -> GraphState:
    pipeline = state["pipeline"]
    user_text = state["query"]
    num_images = len(state["images"])
    
    results = pipeline.run(user_text=user_text, num_images=num_images)

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
    images = state["images"]
    
    visualizer = Visualizer(pipeline.dataset, image_filenames=images)
    visualizer.visualize(layout, copy=copy_result, show_bbox=False)

    return state


from langgraph.graph import END, StateGraph

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

    return app

if __name__ == "__main__":

    # 초기 상태 정의
    initial_state: GraphState = {
        "pipeline": pipeline,
        "query": " 빙그레 초코 우유에 대한 홍보 카드뉴스를 만들거야. 왼쪽 면을 거의 다 차지할 정도로 아주 크게 제목을 적어줘. 오른쪽에는 초코우유 이미지 크게 보여줘. 그리고 그림 아래 설명을 간략히 적어줘.",
        "layout": [],
        "copy": [],
        "images": ["choco1.png"], #, "choco2.png"],
    }


    # config 설정(재귀 최대 횟수, thread_id)
    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})

    app = initialize_graph()

    # 그래프 실행 (스트리밍 출력으로 실행)
    invoke_graph(app, initial_state, config)  # 단순 실행
    # stream_graph(app, initial_state, config)  # 스트리밍 출력
