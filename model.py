import os
from openai import OpenAI
# from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()


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