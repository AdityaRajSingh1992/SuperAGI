import os
import json
from abc import ABC, abstractmethod
from superagi.config.config import get_config
import openai

from typing import Type, List
from pydantic import BaseModel, Field

from superagi.helper.embeddings_creator import Embedding_creator_tool
from superagi.tools.base_tool import BaseTool
from superagi.tools.file.read_file import ReadFileTool
import pandas as pd




class EmbeddingsCreatorSchema(BaseModel):
    file_name: str = Field(...,description="The text file to be converted into embeddings.")

class EmbeddingsCreatorTool(BaseTool):
  name = "Embeddings Generator"
  description = (
        "A tool for reading text from a text file and then transforming text into semantic vectors or embeddings. "
        "Input should be a text file."
  )
  args_schema: Type[BaseModel] = EmbeddingsCreatorSchema

  def _execute(self, file_name: str):

        file_content = ReadFileTool(file_name)
        create_embeddings = Embedding_creator_tool(model)
        embeddings = pd.DataFrame(create_embeddings.get_embeddings(file_content))
        print (embeddings)
        #embeddings.to_csv("embeddings.csv", index=False)
        # root_dir = get_config('RESOURCES_INPUT_ROOT_DIR')
        # final_path = file_name
        # if root_dir is not None:
        #     root_dir = root_dir if root_dir.startswith("/") else os.getcwd() + "/" + root_dir
        #     root_dir = root_dir if root_dir.endswith("/") else root_dir + "/"
        #     final_path = root_dir + file_name
        # else:
        #     final_path = os.getcwd() + "/" + file_name
        #
        # directory = os.path.dirname(final_path)
        # os.makedirs(directory, exist_ok=True)
        #
        # file = open(final_path, 'r')
        # file_content = file.read()
        # return file_content[:1500]
        #

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
