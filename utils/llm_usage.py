import ollama
from ollama import AsyncClient

async def local_llm(content, model_name="deepseek-r1:14b"):
    """
    Generate a response from the specified LLM model.

    Args:
        content (str): The input content to be processed by the model.
        model_name (str): The name of the model to use. Default is "deepseek-r1:14b".

    Returns:
        str: The generated response from the model.
    """
    response = await AsyncClient().chat(
        model=model_name,
        messages=[
            {"role": "user", "content": content},
        ],
    )
    return response["message"]["content"]
