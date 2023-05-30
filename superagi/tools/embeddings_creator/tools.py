import os
import json
from abc import ABC, abstractmethod
from superagi.config.config import get_config
import openai

from typing import Type, List
from pydantic import BaseModel, Field

from superagi.helper.embeddings_creator import Embedding_creator_tool
from superagi.tools.base_tool import BaseTool



class EmbeddingsCreatorSchema(BaseModel):
        text_input: str = Field(
        ...,
        description="The text to be converted into embeddings.",
    )

class EmbeddingsCreatorTool(BaseTool):
  name = "Embeddings Generator"
    description = (
        "A tool for converting text into embeddings"
        "Input should be a text file."
    )
    args_schema: Type[EmbeddingsCreatorSchema] = EmbeddingsCreatorSchema
       
        
       def __init__(self, model="text-embedding-ada-002"):
        self.model = model
        
       def get_embedding(self, text):
        try:
            openai.api_key = get_config("OPENAI_API_KEY")
            print(openai.api_key)
            response = openai.Embedding.create(
                input=[text],
                engine=self.model
            )
            return response['data'][0]['embedding']
        except Exception as exception:
            return {"error": exception} 
        
       
      

# class BaseEmbedding(ABC):

#   @abstractmethod
#   def get_embedding(self, text):
#     pass

  
  
  
# class OpenAiEmbedding:
#     def __init__(self, model="text-embedding-ada-002"):
#         self.model = model

#     async def get_embedding_async(self, text):
#         try:
#             openai.api_key = get_config("OPENAI_API_KEY")
#             response = await openai.Embedding.create(
#                 input=[text],
#                 engine=self.model
#             )
#             return response['data'][0]['embedding']
#         except Exception as exception:
#             return {"error": exception}

#     def get_embedding(self, text):
#         try:
#             openai.api_key = get_config("OPENAI_API_KEY")
#             print(openai.api_key)
#             response = openai.Embedding.create(
#                 input=[text],
#                 engine=self.model
#             )
#             return response['data'][0]['embedding']
#         except Exception as exception:
#             return {"error": exception}
