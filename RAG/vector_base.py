import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.append(cur_dir)

import re
from uuid import uuid4
from embedding import OllamaLocalEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class VectorDataBase:
    def __init__(self, vectordb_name):
        self.chunk_size = 500
        self.overlap_size = 50
        self.vectordb_name = vectordb_name

        self.embedd_model = OllamaLocalEmbeddings(model="mxbai-embed-large:latest")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap_size
        )

        self.vectordb = Chroma(
            collection_name=vectordb_name,
            embedding_function=self.embedd_model,
            persist_directory=vectordb_name
        )

    def add_documents(self, doc_files):
        texts = self.read_files_texts(doc_files)
        split_docs = self.doc_preprocess(texts)
        print(f"切分后的文件数量：{len(split_docs)}")

        uuids = [str(uuid4()) for _ in range(len(split_docs))]
        self.vectordb.add_documents(split_docs, ids=uuids)
        print("数据存储完成")

    def search(self, query, top_n=5, search_type='similarity'):
        if search_type == 'similarity':
            results = self.vectordb.similarity_search_with_score(query, k=top_n)
        elif search_type == 'mmr':
            results = self.vectordb.max_marginal_relevance_search(query, k=top_n)
        else:
            raise ValueError("search_type must be 'similarity' or 'mmr'")
        return results

    def read_files_texts(self, doc_file):
        file_paths = []
        texts = []
        loaders = []

        for root, dirs, files in os.walk(doc_file):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

        for file_path in file_paths:
            file_type = file_path.split('.')[-1]
            loaders.append(PyMuPDFLoader(file_path))

        for loader in loaders: texts.extend(loader.load())
        return texts

    def doc_preprocess(self, doc_pages):
        for page in doc_pages:
            pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
            page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), page.page_content)
            page.page_content = page.page_content.replace('•', '')
            page.page_content = page.page_content.replace(' ', '')
            page.page_content = page.page_content.replace('—', '')
            page.page_content = page.page_content.replace('——', '')
            page.page_content = page.page_content.replace('*', '')
            page.page_content = page.page_content.replace('', '')

        split_docs = self.text_splitter.split_documents(doc_pages)
        return split_docs

if __name__ == '__main__':
    vectordb = VectorDataBase(vectordb_name='vectordb')
    vectordb.add_documents('doc_files/')
    # results = vectordb.search("南瓜书的作者是谁", top_n=2)
    # for i, (res, score) in enumerate(results):
    #     print(f"检索到的第{i}个内容: \nSIM: [SIM={score:3f}] {res.page_content}", end="\n--------------\n")



