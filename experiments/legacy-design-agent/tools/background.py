# DEPRECATED
## This file is deprecated and will be removed in future versions.

from langchain_core.tools import tool
from typing import List, Dict

@tool
def background_gen(query: str) -> List[Dict[str, str]]:
    """Generate a background style based on the query."""
    if query == "gradient":
        return [{"background": "linear-gradient(to right, #FF7E5F, #FEB47B)"}]
    elif query == "pattern":
        return [{"background": "url('pattern.png')"}]
    else:
        return [{"background": "#FFFFFF"}]
