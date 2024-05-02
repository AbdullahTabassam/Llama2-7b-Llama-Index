from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

import torch


system_prompt ="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""

query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")


llm = HuggingFaceLLM(
	context_window = 4096,
	max_new_tokens = 256,
	generate_kwargs = {"temperature" : 0.0, "do_sample" : False},
	system_prompt = system_prompt, 
	query_wrapper_prompt = query_wrapper_prompt,
	model_name = "meta-llama/Llama-2-7b-chat-hf",
	tokenizer_name = "meta-llama/Llama-2-7b-chat-hf", 
	device_map = "auto",
)



documents = SimpleDirectoryReader('files').load_data()

embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2"))

service_context = ServiceContext.from_defaults(
	chunk_size = 1024,	
	llm = llm,
	embed_model = embed_model
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()

print(query_engine.query("what is MLOps?"))
