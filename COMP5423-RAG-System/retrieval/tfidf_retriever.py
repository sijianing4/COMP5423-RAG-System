
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

class TFIDFRetriever:
    """TF-IDF检索器 - 基于你的demo代码"""
    
    def __init__(self, documents: List[str], doc_ids: List[str]):
        """
        初始化TF-IDF检索器
        
        Args:
            documents: 文档内容列表
            doc_ids: 文档ID列表
        """
        self.documents = documents
        self.doc_ids = doc_ids
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        print("正在计算TF-IDF向量...")
        self.doc_vectors = self.vectorizer.fit_transform(documents)
        print("TF-IDF计算完成!")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        检索最相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            List[Dict]: 检索到的文档列表，每个文档包含id, content, score
        """
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
            
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            retrieved_docs = []
            for idx in top_indices:
                retrieved_docs.append({
                    'id': self.doc_ids[idx],
                    'content': self.documents[idx],
                    'score': float(similarities[idx])
                })
            
            return retrieved_docs
        except Exception as e:
            print(f"检索错误: {e}")
            return []
