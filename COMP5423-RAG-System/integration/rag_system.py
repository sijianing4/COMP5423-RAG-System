
import sys
import os
import time
from typing import List, Tuple, Dict, Any

# 添加模块路径
sys.path.append('/content/COMP5423-RAG-System')

from retrieval.tfidf_retriever import TFIDFRetriever
from generation.basic_generator import BasicGenerator
from utils.data_loader import DataLoader

class RAGSystem:
    """主RAG系统 - 整合所有模块"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        """
        初始化RAG系统
        
        Args:
            model_name: 使用的模型名称
        """
        print("🚀 初始化RAG系统...")
        
        # 加载数据
        self.data_loader = DataLoader()
        data = self.data_loader.load_hotpotqa_data()
        
        self.documents = data['documents']
        self.doc_ids = data['doc_ids']
        self.train_df = data['train']
        self.validation_df = data['validation']
        self.collection_df = data['collection']
        
        # 初始化模块
        self.retriever = TFIDFRetriever(self.documents, self.doc_ids)
        self.generator = BasicGenerator(model_name)
        
        print("✅ RAG系统初始化完成")
    
    def rag_pipeline(self, question: str, top_k: int = 10) -> Tuple[str, List[Dict]]:
        """
        完整的RAG流程
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            
        Returns:
            Tuple[str, List[Dict]]: (答案, 检索到的文档列表)
        """
        print(f"\n🎯 用户问题: {question}")
        
        # 步骤1: 检索相关文档
        print("🔍 正在检索相关文档...")
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
        
        if not retrieved_docs:
            return "未找到相关文档", []
        
        print("📄 检索到的文档:")
        for i, doc in enumerate(retrieved_docs):
            print(f"  {i+1}. [ID: {doc['id']}, 相似度: {doc['score']:.4f}]")
            print(f"     内容: {doc['content'][:150]}...")
        
        # 步骤2: 生成答案
        print("💭 正在生成答案...")
        answer = self.generator.generate_answer(question, retrieved_docs)
        
        return answer, retrieved_docs
    
    def rag_interface(self, question: str, top_k: int = 10) -> Tuple[str, str, List[Tuple]]:
        """
        供界面调用的RAG函数
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            
        Returns:
            Tuple[str, str, List[Tuple]]: (答案, 统计信息, 文档列表)
        """
        if not question.strip():
            return "请输入问题", "", []
        
        start_time = time.time()
        
        try:
            # 1. 检索文档
            retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
            retrieval_time = time.time() - start_time
            
            if not retrieved_docs:
                return "未找到相关文档", f"检索时间: {retrieval_time:.2f}s | 找到0个文档", []
            
            # 2. 生成答案
            generation_start = time.time()
            answer = self.generator.generate_answer(question, retrieved_docs)
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            
            # 3. 构建统计信息
            stats = (f"⏱️ 总时间: {total_time:.2f}s | "
                    f"🔍 检索: {retrieval_time:.2f}s | "
                    f"🤖 生成: {generation_time:.2f}s | "
                    f"📄 文档: {len(retrieved_docs)}个")
            
            # 4. 格式化文档信息供界面显示
            doc_display = []
            for i, doc in enumerate(retrieved_docs):
                doc_display.append((
                    f"文档 {i+1}",
                    f"ID: {doc['id']}\n相似度: {doc['score']:.4f}\n内容: {doc['content'][:200]}..."
                ))
            
            return answer, stats, doc_display
            
        except Exception as e:
            return f"处理错误: {str(e)}", "", []
    
    def interactive_demo(self):
        """交互式演示"""
        print("\n" + "="*60)
        print("🚀 HotpotQA RAG Demo 开始!")
        print("="*60)
        
        # 从训练集中提取示例问题
        test_questions = self.data_loader.get_sample_questions(3, 'train')
        
        if not test_questions:
            test_questions = [
                "What is the capital of France?",
                "Who wrote the novel 'Pride and Prejudice'?",
                "When was the first computer invented?"
            ]
        
        print("🧪 测试问题示例:")
        for i, question in enumerate(test_questions):
            print(f"正在测试问题 {i+1}: {question}")
            answer, docs = self.rag_pipeline(question)
            print(f"💡 生成的答案: {answer}")
            print("-" * 80)
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'model_name': self.generator.model_name,
            'document_count': len(self.documents),
            'train_samples': len(self.train_df),
            'validation_samples': len(self.validation_df),
            'retrieval_method': 'TF-IDF'
        }

if __name__ == "__main__":
    # 测试系统
    rag_system = RAGSystem()
    rag_system.interactive_demo()
