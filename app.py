from flask import Flask
import os 
from urllib.parse import parse_qs
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI


os.environ['OPENAI_API_KEY'] = "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"
def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 2000
    max_chunk_overlap = 20
    chunk_size_limit = 600
    max_tokens=500
    prompt_helper = PromptHelper(max_input_size, max_tokens, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=max_tokens))
    documents = SimpleDirectoryReader(directory_path).load_data()
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    return index

index=construct_index("data")
query_engine = index.as_query_engine()



app = Flask(__name__)
@app.route('/<name>')
def hello(name):
    name = query_engine.query(name)
    return f"bot: {name}!"


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')