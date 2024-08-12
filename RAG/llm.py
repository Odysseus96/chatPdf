from utils import *
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

class RAGChatBot:
    def __init__(self, vector_db, llm, n_results=2):
        self.vector_db = vector_db
        self.llm = llm
        self.n_results = n_results

        self.prompt_template = PromptTemplate.from_file("prompt/prompt_internlm2.txt")

        self.qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever = self.vector_db.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )

        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            prompt=self.prompt_template,
        )

        self.chain = create_retrieval_chain(self.vector_db.as_retriever(), question_answer_chain)

    def chat(self, question):
        res = self.chain.stream({'input': question})
        # response = self.llm.stream(prompt)
        return res


if __name__ == '__main__':
    from vector_base import VectorDataBase
    vector_base = VectorDataBase('vectordb')
    llm = ChatOllama(model='internlm2:7b-chat-v2.5-q4_K_M', temperature = 0.7)

    question = "RAG是什么？"

    print("******* LLM Answer ********\n")
    stream = llm.stream(question)
    for chunk in stream:
        print(chunk.content, end='', flush=True)

    print("\n******* LLM+知识库 Answer  ********\n")
    rag_chatbot = RAGChatBot(vector_db=vector_base.vectordb, llm=llm)
    results = rag_chatbot.chat(question)
    for result in results:
        if answer_chunk := result.get("answer"):
            print(answer_chunk, end='', flush=True)
    print('\n')
