import os
import openai

import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt

class Embedding_creator_tool:
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
        return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]
