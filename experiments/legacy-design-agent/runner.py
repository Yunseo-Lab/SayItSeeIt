import webbrowser
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from nodes import State
from utils import generate_and_save_html
from graph_builder import build_graph

# Runner setup
def main(canvas_size: tuple, question: str, open_browser: bool = True):
    graph = build_graph()
    config = RunnableConfig(recursion_limit=10, configurable={"thread_id": "1"})

    # Initialize state
    initial_state = {  
        "canvas_size": canvas_size,  # 캔버스 크기 초기값 설정  
        "messages": [("user", question)]  # 사용자 질문으로 메시지 초기화  
    }  

    # Stream and display messages
    for event in graph.stream(initial_state, config=config):
        for value in event.values():
            value["messages"][-1].pretty_print()

    # Final output
    state = graph.get_state({"configurable": {"thread_id": "1"}})
    final_output = state.values["messages"][-1].content
    
    # Save & open HTML
    generate_and_save_html(final_output, open_browser=open_browser)

    # Print final output
    print(state.values)



if __name__ == "__main__":
    # Example usage
    canvas_size = (800, 600) # width, height
    query = "옥수수 수염차에 대한 홍보 포스터 만들어줘."
    main(canvas_size, query, open_browser=True)
