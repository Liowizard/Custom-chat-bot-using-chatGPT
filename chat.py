import os 
import json
from urllib.parse import parse_qs
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import sys
import os
from IPython.display import Markdown, display



os.environ['OPENAI_API_KEY'] = "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"



def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 2000
    max_chunk_overlap = 20
    chunk_size_limit = 600
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data()
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    return index



index=construct_index("data")
query_engine = index.as_query_engine()



def parse_query_params(query_string):
    params = parse_qs(query_string)
    return {key: value[0] for key, value in params.items()}



def hello_world(environ, start_response):
    query_string = environ.get('QUERY_STRING', '')
    query_params = parse_query_params(query_string)
    name = query_params.get('prompt')
    print(name)
    status = '200 OK'
    response_headers = [('Content-type', 'application/json')]
    start_response(status, response_headers)

    name = query_engine.query(name)

    if name:
        response = {'message': f'Hello, {name}!'}
    # else:
    #     response = {'message': 'Hello, World!'}
    return [json.dumps(response).encode('utf-8')]



if __name__ == '__main__':
    from wsgiref.simple_server import make_server
    httpd = make_server('', 8000, hello_world)
    print("Serving on port 8000...")
    httpd.serve_forever()
