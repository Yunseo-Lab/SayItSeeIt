from langchain_core.tools import tool
from typing import List, Dict

@tool
def color_gen(query: str) -> List[Dict[str, str]]:
    """Generate a list of colors based on the query."""
    if query == "warm":
        return [{"red": "#FF0000"}, {"orange": "#FFA500"}, {"yellow": "#FFFF00"}]
    return [{"blue": "#0000FF"}, {"green": "#008000"}, {"purple": "#800080"}]
