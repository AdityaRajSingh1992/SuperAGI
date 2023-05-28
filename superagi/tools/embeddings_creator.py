from typing import Type
from pydantic import BaseModel, Field

from superagi.helper.embedding_creator import GoogleSerpApiWrap
from superagi.tools.base_tool import BaseTool
from superagi.config.config import get_config


import os


import json

class EmbeddingToolSchema(BaseModel):
    query: str = Field(
        ...,
        description="The text to embedding converter.",
    )


class EmbeddingTool(BaseTool):
    name = "Embedding Creator"
    description = (
        "A tool for performing a text to embedding convertor"
        "Input should be a text writeup."
    )
    args_schema: Type[EmbeddingToolSchema] = EmbeddingToolSchema

    def _execute(self, query: str) -> tuple:
        api_key = get_config("OPENAI_API_KEY")
        #num_results = 10
        #num_pages = 1
        #num_extracts = 3

        embedding_creator = Embedding_creator_tool(api_key)#,num_results, num_pages, num_extracts)
        return embedding_creator.get_embedding(text)
      
      
     

