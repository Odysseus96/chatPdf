import gradio as gr

from RAG.vector_base import VectorDataBase
from RAG.llm import RAGChatBot
from langchain_ollama import ChatOllama

MODEL_NAME = 'qwen2:7b'


def format_history(msg: str, history: list[list[str, str]], system_prompt: str):
    chat_history = [{"role": "system", "content":system_prompt}]
    for query, response in history:
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})  
    chat_history.append({"role": "user", "content": msg})
    return chat_history


def generate_response(msg: str, history: list[list[str, str]]):
    llm = ChatOllama(model=MODEL_NAME, temperature=0.2)
    vector_base = VectorDataBase('vectordb')

    chatbot = RAGChatBot(vector_base.vectordb, llm, 2)
    response = chatbot.chat(msg)
    message = ""
    for chunk in response:
        if answer_cunk := chunk.get("answer"):
            message += answer_cunk
            yield message


chatbot = gr.ChatInterface(
                generate_response,
                # additional_inputs=[
                #     gr.Textbox(
                #         "Behave as if you are professional writer.",
                #         label="System Prompt"
                #     )
                # ],
                title="深度学习小助手",
                description="帮助解答深度学习相关的问题。",
                theme="soft",
                submit_btn="⬅ Send",
                retry_btn="🔄 Regenerate Response",
                undo_btn="↩ Delete Previous",
                clear_btn="🗑️ Clear Chat"
)

# chatbot.launch(server_name='0.0.0.0', server_port=8080)
chatbot.launch(share=True)
