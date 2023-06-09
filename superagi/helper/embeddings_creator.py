import os
import openai

import openai

from superagi.config.config import get_config


#from tenacity import retry, wait_random_exponential, stop_after_attempt

class Embedding_creator_tool:

    # def __init__(self):
    #     self.model

    #@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embeddings(self, text: str):
        try:
            model = "text-embedding-ada-002"
            openai.api_key = get_config("OPENAI_API_KEY")
            print(openai.api_key)
            response = openai.Embedding.create(
                input=[text],
                engine=model)
            return response['data'][0]['embedding']

        except Exception as exception:
            return {"error": exception}
