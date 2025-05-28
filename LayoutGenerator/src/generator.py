from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()


def generate_layout(prompt: str):

    responses = []
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
            "role": "system",
            "content": [
                {
                "type": "input_text",
                "text": prompt
                }
            ]
            }
        ],
        text={
            "format": {
            "type": "text"
            }
        },
        reasoning={},
        tools=[],
        temperature=1,
        max_output_tokens=2048,
        top_p=1,
        store=False
    ) 
    responses.append(response.output_text)

    return responses

if __name__ == "__main__":
    user_prompt = "Create a layout for a blog post about AI advancements in 2023."
    layout = generate_layout(user_prompt)
    print("Generated Layout:")
    print(layout)

