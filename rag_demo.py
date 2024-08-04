from vector_base import VectorStore
from utils import ReadFiles
from llm import OllamaChat
from embedding import OllamaEmbedding


# 没有保存数据库
docs = ReadFiles('./Users/wangyaozhong/CodeForFuture/papers/llama2.pdf').get_content(max_token_len=600, cover_content=150) # 获得data目录下的所有文件内容并分割
vector = VectorStore(docs)
embedding = OllamaEmbedding() # 创建EmbeddingModel
vector.get_vector(embeddingModel=embedding)
vector.persist(path='storage') # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库

question = 'git的原理是什么？'

content = vector.query(question, embeddingModel=embedding, k=1)[0]
chat = OllamaChat('qwen:7b')
print(chat.chat(question, [], content))