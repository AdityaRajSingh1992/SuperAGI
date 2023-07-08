import json
from abc import ABC, abstractmethod
from superagi.config.config import get_config
import openai

from typing import Type, List
from pydantic import BaseModel, Field

from superagi.helper.knowledge_tool import KnowledgeToolHelper
from superagi.models.agent_config import AgentConfiguration
from superagi.models.knowledge import Knowledge
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
    agent_id: int = None
    description = (
        "A tool for performing a Knowledge search on knowledge base which might have knowledge of the task you are pursuing."
        "To find relevant info, use this tool first before using other tools."
        "If you don't find sufficient info using Knowledge tool, you may use other tools."
        "If a question is being asked, responding with context from info returned by knowledge tool is prefered."
        "Input should be a search query."
    )

    def _execute(self, query: str):
        print(query)
        openai_api_key = get_config("OPENAI_API_KEY")
        knowledge_base = get_config("KNOWLEDGE_BASE")  #: QDRANT #PINECONE/CHROMA/WEAVIATE
        knowledge_api_key = get_config("KNOWLEDGE_API_KEY")
        knowledge_index_or_collection = get_config("KNOWLEDGE_INDEX_OR_COLLECTION")
        knowledge_url = get_config("KNOWLEDGE_URL")
        knowledge_environment = get_config("KNOWLEDGE_ENVIRONMENT")
        knowledge_ids = self.toolkit_config.session.query(AgentConfiguration).filter(
            AgentConfiguration.agent_id == self.agent_id,
            AgentConfiguration.key == "knowledge").first()
        knowledge_names = []
        for knowledge_id in knowledge_ids:
            knowledge = self.toolkit_config.session.query(Knowledge).filter(Knowledge.id == knowledge_id).first()
            if knowledge:
                knowledge_names.append(knowledge.name)

        query_knowledge = KnowledgeToolHelper(openai_api_key, knowledge_api_key, knowledge_index_or_collection,
                                              knowledge_url, knowledge_environment,knowledge_names)
        if knowledge_base == 'PINECONE':
            req_context = query_knowledge.pinecone_get_match_vectors(query)
        elif knowledge_base == 'QDRANT':
            req_context = query_knowledge.qdrant_get_match_vectors(query)
        elif knowledge_base == 'CHROMA':
            req_context = query_knowledge.chroma_get_match_vectors(query)
        return req_context
