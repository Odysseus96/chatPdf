import os
from typing import Dict, List, Optional, Tuple, Union
import ollama
from ollama import Client

PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
    InternLM_PROMPT_TEMPALTE="""先对上下文进行内容总结,再使用上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
)

BASE_URL = 'http://localhost:8181'

class BaseModel:
    def __init__(self, model):
        self.model = model

    def chat(self, prompt, history, content):
        pass

    def load_model(self):
        pass


class OllamaChat(BaseModel):
    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.url = BASE_URL
        self.client = Client(host=BASE_URL)

    def chat(self, prompt, history, content):
        history.append({'role': 'user', 'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)})
        response = self.client.chat(self.model, messages=history)
        return response['message']['content']