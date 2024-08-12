from RAG.embedding import OllamaLocalEmbeddings
from RAG.utils import *

query = "国际争端"

documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间站开展舱外辐射生物学暴露实验",
]

embed_model_name = "mxbai-embed-large"
embedding_model = OllamaLocalEmbeddings(model=embed_model_name)

query_vec = embedding_model.get_embeddings(query)
print(len(query_vec))

doc_vecs = [
    embedding_model.get_embeddings(doc)
    for doc in documents
]

print(f"{embed_model_name} Cosine similarity:")
for vec in doc_vecs:
    print(cosine_similary(query_vec, vec))



