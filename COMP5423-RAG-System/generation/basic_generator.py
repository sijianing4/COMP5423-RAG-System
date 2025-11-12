
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict

class BasicGenerator:
    """基础生成器 - 基于你的demo代码"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        """
        初始化生成器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        print(f"🤖 加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ 模型加载完成")
    
    def generate_answer(self, question: str, retrieved_docs: List[Dict]) -> str:
        """
        使用检索到的文档生成答案
        
        Args:
            question: 用户问题
            retrieved_docs: 检索到的文档列表
            
        Returns:
            str: 生成的答案
        """
        # 构建提示模板
        context = "\n".join([f"[文档 {i+1}, ID: {doc['id']}]: {doc['content']}"
                            for i, doc in enumerate(retrieved_docs)])

        prompt = f"""你是一个智能问答助手。请基于以下提供的文档内容，准确回答用户的问题。只使用文档中的信息，不要编造内容。

相关文档：
{context}

用户问题：{question}

请基于上述文档提供准确、简洁的答案。如果文档中没有相关信息，请明确说明"根据提供的文档，无法回答这个问题"。

答案："""

        # 准备模型输入
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码输出
        response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return response.strip()
    
    def batch_generate(self, questions: List[str], all_retrieved_docs: List[List[Dict]]) -> List[str]:
        """
        批量生成答案
        
        Args:
            questions: 问题列表
            all_retrieved_docs: 每个问题对应的检索文档列表
            
        Returns:
            List[str]: 答案列表
        """
        answers = []
        for question, retrieved_docs in zip(questions, all_retrieved_docs):
            answer = self.generate_answer(question, retrieved_docs)
            answers.append(answer)
        return answers
