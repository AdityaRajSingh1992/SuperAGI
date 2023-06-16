import json
from abc import ABC, abstractmethod
from superagi.config.config import get_config
import openai

from typing import Type, List
from pydantic import BaseModel, Field

from superagi.helper.knowledge_tool import Knowledgetoolhelper
from superagi.tools.base_tool import BaseTool
# from superagi.tools.file.read_file import ReadFileTool
import pandas as pd

# 1. define input schema
class KnowledgeSearchSchema(BaseModel):
    query: str = Field(..., description="The search query for knowledge store search")

# 2. setup name, arg, description
class KnowledgeSearchTool(BaseTool):
    name: str = "Knowledge Search"
    args_schema: Type[BaseModel] = KnowledgeSearchSchema
    description = (
        "A tool for performing a Knowledge search on knowledge base which might have knowledge of the task you are pursuing."
        "To find relevant info, use this tool first before using other tools."
        "If you don't find sufficient info using Knowledge tool, you may use other tools."
        "If a question is being asked, responding with context from info returned by knowledge tool is prefered."
        "Input should be a search query."
    )

    def _execute(self, query: str):
        print(query)
        query_knowledge = Knowledgetoolhelper()
        req_context = query_knowledge.get_match_vectors(query)
        return req_context
