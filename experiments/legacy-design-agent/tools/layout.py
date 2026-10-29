# DEPRECATED: This tool is deprecated and will be removed in future versions.
# Please use the new tool system for layout generation.


from langchain_core.tools import tool
from typing import List, Dict
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from utils import load_system_template

from dotenv import load_dotenv

load_dotenv()

@tool
def layout_gen(query: str) -> List[Dict[str, list]]:
    """Generate a layout style based on the query."""
    # Use an LLM to generate a layout based on the query

    # Load your system template
    system_template = load_system_template("/Users/localgroup/Documents/workspace/say_it_see_it/prompts/layout_prompt.yaml")

    # Prepare the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
        ('system', system_template),
        MessagesPlaceholder(variable_name='messages'),
        ]
    )
    llm = ChatOpenAI(model='gpt-4o-mini')
    chain = prompt | llm

    # Invoke the chain with the query as a message
    result = chain.invoke({'messages': [{'role': 'user', 'content': query}]})

    # Extract and return the layout from the LLM response
    return result
    
if __name__ == "__main__":
    # Example usage
    example_query = ""
    layout = layout_gen(example_query)
    print("Generated layout:", layout)