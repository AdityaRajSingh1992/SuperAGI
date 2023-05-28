import os
import openai
import langchain 

# from langchain.embeddings.openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(
#     deployment="your-embeddings-deployment-name",
#     model="your-embeddings-model-name",
#     api_base="https://your-endpoint.openai.azure.com/",
#     api_type="azure",
# )
# text = "This is a test query."
# query_result = embeddings.embed_query(text)

# def get_embedding(text, model="text-embedding-ada-002"):
#    text = text.replace("\n", " ")
#    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
# df.to_csv('output/embedded_1k_reviews.csv', index=False)

import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt

class Embedding_creator_tool:
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]
