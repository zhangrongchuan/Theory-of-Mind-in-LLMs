import os
from openai import OpenAI
# from huggingface_hub import InferenceClient
from dotenv import load_dotenv
# from ollama import chat
load_dotenv()

def call_model_ollama(prompt: str) -> str:
    """
    Call an Ollama model to get the answer.
    model: e.g. "qwen:4b", "qwen3:8b", etc.
    """

    response = chat(
        model='qwen:4b',  # change model if needed
        messages=[
            #{"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
    )

    return response.message.content

def call_model_ollama_SoO(prompt: str) -> str:
    """
    Call an Ollama model to get the answer.
    model: e.g. "qwen:4b", "qwen3:8b", etc.
    """

    response = chat(
        model='qwen:4b',  # change model if needed
        messages=[
            {"role": "system", "content": "You are an expert at understanding human communication. Please leverage the Information provided and choose the most probable answer to the question from the options. Output your final answer by strictly following this format: [A], [B], [C], or [D]"},
            {"role": "user", "content": prompt},
        ],
    )

    return response.message.content


def call_model_deepseek(prompt: str) -> str:
    """
    call deepseek API to get the answer.
    model: "deepseek-chat" (DeepSeek-V3.2) or "deepseek-reasoner" (DeepSeek-V3.2 thinking mode)
    """

    # for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
    client = OpenAI(api_key=os.getenv("deepseek_api"), base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
    ],
        max_tokens=1024,
        temperature=0.0,
        stream=False
    )

    # print(response.choices[0].message.content)
    return response.choices[0].message.content
