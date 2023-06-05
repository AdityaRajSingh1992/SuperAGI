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
  args_schema: Type[BaseModel] = EmbeddingsCreatorSchema
  description = (
        "A tool for reading text from a text file and then transforming text into semantic vectors or embeddings. "
        "Input should be a text file."
  )
  
  def _execute(self, file_name: str):
        model = "text-embedding-ada-002"
        file_content = ReadFileTool(file_name)
        create_embeddings = Embedding_creator_tool(model)
        embeddings = pd.DataFrame(create_embeddings.get_embeddings(file_content))
        print (embeddings)
 
