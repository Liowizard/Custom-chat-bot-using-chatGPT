import json
from urllib.parse import parse_qs
import os 
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import sys
import os
from IPython.display import Markdown, display

os.environ['OPENAI_API_KEY'] = "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"
  



def parse_query_params(query_string):
    params = parse_qs(query_string)
    return {key: value[0] for key, value in params.items()}
def hello_world(environ, start_response):
    query_string = environ.get('QUERY_STRING', '')
    query_params = parse_query_params(query_string)
    name = query_params.get('prompt')
    status = '200 OK'
    response_headers = [('Content-type', 'application/json')]
    start_response(status, response_headers)


    def construct_index(directory_path):
        # set maximum input size
        max_input_size = 4096
        # set number of output tokens
        num_outputs = 2000
        # set maximum chunk overlap
        max_chunk_overlap = 20
        # set chunk size limit
        chunk_size_limit = 600 

        # define prompt helper
        prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

        # define LLM
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
    
        documents = SimpleDirectoryReader(directory_path).load_data()
        
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

        #index.save_to_disk('index.json')

        return index
    
    index=construct_index("data")

    def ask_ai(name): 
        query_engine = index.as_query_engine()
        response = query_engine.query(name)
        return response

    



    name=ask_ai(name)
    if name:
        response = {'message': name}
    else:
        response = {'message': 'Hello, World!'}
    return [json.dumps(response).encode('utf-8')]
if __name__ == '__main__':
    from wsgiref.simple_server import make_server
    httpd = make_server('', 8000, hello_world)
    print("Serving on port 8000...")
    httpd.serve_forever()