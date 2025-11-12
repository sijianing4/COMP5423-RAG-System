
from google.colab import drive
import pandas as pd
from typing import Dict, Tuple, List
import os

class DataLoader:
    """数据加载器 - 处理Google Drive中的数据文件"""
    
    def __init__(self):
        self.drive_mounted = False
        self.data_loaded = False
        self.train_df = None
        self.validation_df = None
        self.collection_df = None
        self.documents = []
        self.doc_ids = []
    
    def mount_drive(self) -> bool:
        """挂载Google Drive"""
        if not self.drive_mounted:
            try:
                drive.mount('/content/drive')
                self.drive_mounted = True
                print("✅ Google Drive挂载完成")
                return True
            except Exception as e:
                print(f"❌ Google Drive挂载失败: {e}")
                return False
        return True
    
    def load_hotpotqa_data(self, base_path: str = '/content/drive/MyDrive/RAGtest') -> Dict:
        """
        加载HotpotQA数据集
        
        Args:
            base_path: 数据文件基础路径
            
        Returns:
            Dict: 包含所有数据的字典
        """
        if not self.mount_drive():
            raise Exception("无法挂载Google Drive")
        
        print("📚 加载数据...")
        try:
            self.train_df = pd.read_json(f'{base_path}/train.jsonl', lines=True)
            self.validation_df = pd.read_json(f'{base_path}/validation.jsonl', lines=True)
            self.collection_df = pd.read_json(f'{base_path}/collection.jsonl', lines=True)
            
            print(f"训练集: {len(self.train_df)} 样本")
            print(f"验证集: {len(self.validation_df)} 样本")
            print(f"文档集: {len(self.collection_df)} 文档")
            
            # 准备文档集合
            self.documents = []
            self.doc_ids = []
            for idx, row in self.collection_df.iterrows():
                if 'id' in row and 'text' in row:
                    self.doc_ids.append(row['id'])
                    self.documents.append(row['text'])
            
            print(f"文档库大小: {len(self.documents)} 个文档")
            self.data_loaded = True
            
            return {
                'train': self.train_df,
                'validation': self.validation_df,
                'collection': self.collection_df,
                'documents': self.documents,
                'doc_ids': self.doc_ids
            }
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise
    
    def get_sample_questions(self, num_samples: int = 3, split: str = 'train') -> List[str]:
        """
        获取示例问题
        
        Args:
            num_samples: 样本数量
            split: 数据集分割 (train/validation)
            
        Returns:
            List[str]: 示例问题列表
        """
        if not self.data_loaded:
            self.load_hotpotqa_data()
        
        df = self.train_df if split == 'train' else self.validation_df
        questions = []
        
        for i in range(min(num_samples, len(df))):
            if 'question' in df.columns:
                questions.append(df['question'].iloc[i])
            elif 'text' in df.columns:
                questions.append(df['text'].iloc[i])
        
        return questions
    
    def get_data_info(self) -> Dict:
        """获取数据信息统计"""
        if not self.data_loaded:
            self.load_hotpotqa_data()
        
        return {
            'train_samples': len(self.train_df),
            'validation_samples': len(self.validation_df),
            'collection_documents': len(self.collection_df),
            'train_columns': self.train_df.columns.tolist(),
            'validation_columns': self.validation_df.columns.tolist(),
            'collection_columns': self.collection_df.columns.tolist()
        }
