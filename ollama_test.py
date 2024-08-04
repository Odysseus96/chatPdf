import ollama
from ollama import Client
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage


BASE_URL = 'http://localhost:8181'
client = Client(host=BASE_URL)

llm = Ollama(model='qwen:7b', base_url=BASE_URL)

prompt = 'RAG是检索增强生成（Retrieval-Augmented Generation），是一种结合了检索（Retrieval）和生成（Generation）两种技术的人工智能方法。它主要用于提升大型语言模型（Large Language Models, LLMs）在处理知识密集型任务时的性能。'

messages = [
    ChatMessage(
        role='system', content = prompt
    ),
    ChatMessage(role='user', content='如何实现一个RAG应用?')
]

response = llm.stream_chat(messages)

for chunk in response:
  print(chunk.delta, end='', flush=True)

# messages = [
#     {'role':'system', 'content':prompt},
#     {'role':'user', 'content':'如何实现一个RAG应用'}
# ]
#
# response = client.chat(model='qwen:7b', messages=messages, stream=True)
#
# for chunk in response:
#   print(chunk['message']['content'], end='', flush=True)


