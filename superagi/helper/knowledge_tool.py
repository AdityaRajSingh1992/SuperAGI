import pinecone
import openai
from superagi.config.config import get_config
# def get_search_index(self):
#     embed_model = "text-embedding-ada-002"
#     index_name = PineconeSearchModel.search_index_name
#     pinecone.init(api_key="b5023255-b521-4ba6-aefc-f55d65f5a59b", enviroment="us-east-1-aws")
#     if index_name not in pinecone.list_indexes():
#       openai.api_key = os.getenv("OPENAI_API_KEY")
#       res = openai.Embedding.create(
#         input=[
#           "Sample document text goes here",
#           "there will be several phrases in each batch"
#         ], engine=embed_model
#       )

#       # if does not exist, create index
#       pinecone.create_index(
#         index_name,
#         dimension=len(res['data'][0]['embedding']),
#         metric='dotproduct'
#       )
#     # # connect to index
#     index = pinecone.Index(index_name)
#     return index

#self.search_index_name)
class Knowledgetoolhelper:
  def __init__(self,openai_api_key,pinecone_api_key, pinecone_environment):
    self.openai_api_key = openai_api_key
    self.pinecone_api_key = pinecone_api_key
    self.pinecone_environment = pinecone_environment
    
  def get_match_vectors(self, query):
    print(get_config('PINECONE_API_KEY'), get_config('PINECONE_ENVIRONMENT'))
    #pinecone.init(api_key=get_config('PINECONE_API_KEY'), environment=get_config('PINECONE_ENVIRONMENT'))
    embed_model = "text-embedding-ada-002"
    namespace = ""
    #namespace = "SEO Success_lang"
    # print(pinecone.list_indexes())
    pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_environment)
    index = pinecone.Index('knowledge') 

    #openai.api_key = get_config('OPENAI_API_KEY')
#   t1_start = perf_counter()
    query_res = openai.Embedding.create(
      input=[query],
      engine=embed_model
    )
#   t1_stop = perf_counter()
#   print("OpenAI Elapsed time:", t1_stop - t1_start)
    # retrieve from Pinecone
    x_query = query_res['data'][0]['embedding']
    # get relevant contexts (including the questions)
#    search_res = index.query(x_query, top_k=5, namespace=namespace, include_metadata=True)#, include_values=True)
    search_res = index.query(x_query, top_k=5, include_metadata=True)#, include_values=True)
#   t1_stop2 = perf_counter()
#   print("Pinecone Elapsed time:", t1_stop2 - t1_stop)
    print(search_res)
    contexts = [item['metadata']['text'] for item in search_res['matches']]
    search_res_appended=''
    search_res_appended+=f"\nQuery:{query}\n"
    i=0
    for context in contexts:
      search_res_appended+=str(f'\nchuck{i}:\n')
      search_res_appended+=context
      i+=1
      #print(search_res_appended)

    return search_res_appended
