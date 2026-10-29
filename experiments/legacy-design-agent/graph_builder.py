from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from nodes import State, background_node, layout_node, designer, drawer
from nodes.common import designer_tools, drawer_tools

# Tool nodes
designer_tool_node = ToolNode(tools=designer_tools)
drawer_tool_node = ToolNode(tools=drawer_tools)

# Graph construction
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("designer", designer)
graph_builder.add_node("background", background_node)
graph_builder.add_node("layout", layout_node)
graph_builder.add_node("drawer", drawer)
graph_builder.add_node("designer_tool", designer_tool_node)
graph_builder.add_node("drawer_tool", drawer_tool_node)

# Add edges
graph_builder.add_edge(START, "designer")
graph_builder.add_conditional_edges(
    "designer",
    tools_condition,
    {"tools": "designer_tool", "__end__": "layout"},
)
graph_builder.add_edge("designer_tool", "designer")

graph_builder.add_edge("layout", "background")
graph_builder.add_edge("background", "drawer")
graph_builder.add_conditional_edges(
    "drawer",
    tools_condition,
    {"tools": "drawer_tool", "__end__": END},
)
graph_builder.add_edge("drawer_tool", "drawer")
graph_builder.add_edge("drawer", END)

# Compile graph function
def build_graph():
    memory = MemorySaver()
    
    return graph_builder.compile(checkpointer=memory)
