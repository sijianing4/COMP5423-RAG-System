
#!/usr/bin/env python3
"""
COMP5423 RAG System - 主运行文件
基于HotpotQA数据集的检索增强生成系统
"""

import sys
import os

# 添加模块路径
sys.path.append('/content/COMP5423-RAG-System')

def main():
    """主函数"""
    print("🚀 COMP5423 RAG System")
    print("=" * 50)
    
    # 选择运行模式
    print("请选择运行模式:")
    print("1. 命令行演示模式")
    print("2. Web界面模式")
    print("3. 系统测试模式")
    
    try:
        choice = input("请输入选择 (1/2/3, 默认2): ").strip()
        if not choice:
            choice = "2"
    except:
        choice = "2"
    
    if choice == "1":
        # 命令行演示模式
        from integration.rag_system import RAGSystem
        rag_system = RAGSystem()
        rag_system.interactive_demo()
    
    elif choice == "2":
        # Web界面模式
        from integration.gradio_ui import GradioInterface
        interface = GradioInterface()
        interface.launch()
    
    elif choice == "3":
        # 系统测试模式
        from integration.rag_system import RAGSystem
        from utils.data_loader import DataLoader
        
        print("🧪 系统测试模式...")
        
        # 测试数据加载
        data_loader = DataLoader()
        data_info = data_loader.get_data_info()
        print("数据信息:", data_info)
        
        # 测试RAG系统
        rag_system = RAGSystem()
        test_question = "What is the capital of France?"
        answer, docs = rag_system.rag_pipeline(test_question, top_k=3)
        print(f"测试问题: {test_question}")
        print(f"测试答案: {answer}")
        print(f"检索文档数: {len(docs)}")
        
        print("✅ 系统测试完成")
    
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()
