import os
from typing import Dict, List, Optional, Tuple, Union
import json
import re
import jieba
from nltk.corpus import stopwords
from langchain.prompts import PromptTemplate
import numpy as np


# enc = tiktoken.get_encoding("cl100k_base")
# nltk.download('stopwords')


class Documents:
    """
        获取已分好类的json格式文档
    """

    def __init__(self, path: str = '') -> None:
        self.path = path

    def get_content(self):
        with open(self.path, mode='r', encoding='utf-8') as f:
            content = json.load(f)
        return content


def to_keywords(input_string):
    """将句子转成检索关键词序列"""
    word_tokens = jieba.cut_for_search(input_string)
    # 加载停用词表
    stop_words = stopwords.words('chinese')
    # 去除停用词
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)


def sent_tokenize(input_string):
    """按标点断句"""
    # 按标点切分
    sentences = re.split(r'(?<=[。！？；?!])', input_string)
    # 去掉空字符串
    return [sentence for sentence in sentences if sentence.strip()]


def build_prompt(prompt_file, **kwargs):
    '''将 Prompt 模板赋值'''
    prompt_template = PromptTemplate.from_file(prompt_file)
    return prompt_template.format(**kwargs)


def cosine_similary(a, b):
    '''余弦距离 -- 越大越相似'''
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == '__main__':
    # # 关键词提取
    # print(to_keywords("长期做有价值的事情，这是一句口号"))
    # # 断句测试
    # print(sent_tokenize("这是，第一句。这是第二句吗？是的！啊"))
    context = "长期做有价值的事情"
    query = "长期做什么"
    prompt = build_prompt("prompt.txt", context=context, query=query)
    print(prompt)