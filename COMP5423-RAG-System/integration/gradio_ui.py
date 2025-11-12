
import gradio as gr
import sys
import os

# 添加模块路径
sys.path.append('/content/COMP5423-RAG-System')

from integration.rag_system import RAGSystem

class GradioInterface:
    """Gradio用户界面"""
    
    def __init__(self):
        print("🎨 初始化Gradio界面...")
        self.rag_system = RAGSystem()
        self.demo = self.create_interface()
    
    def create_interface(self):
        """创建Gradio用户界面"""
        
        # CSS样式
        css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .title {
            text-align: center;
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em !important;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .description {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .stats {
            background: #f0f8ff;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #4ECDC4;
            margin: 10px 0;
        }
        .answer-box {
            background: #f8fff0;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #FFD93D;
            margin: 15px 0;
        }
        .doc-box {
            background: #fff0f5;
            padding: 15px;
            border-radius: 8px;
            margin: 8px 0;
            border-left: 3px solid #FF6B6B;
        }
        """
        
        # 获取系统信息
        system_info = self.rag_system.get_system_info()
        
        # 界面布局
        with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
            
            # 标题区域
            gr.Markdown(f"""
            <div class="title">🔍🤖 COMP5423 RAG 智能问答系统</div>
            <div class="description">
            基于HotpotQA数据集的多跳推理问答系统 | 检索增强生成 (Retrieval-Augmented Generation)
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # 输入区域
                    with gr.Group():
                        gr.Markdown("### 💬 输入问题")
                        question_input = gr.Textbox(
                            label="请输入您的问题",
                            placeholder="例如: Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?",
                            lines=3,
                            max_lines=5
                        )
                        
                        with gr.Row():
                            top_k_slider = gr.Slider(
                                minimum=1, maximum=15, value=10, step=1,
                                label="检索文档数量"
                            )
                            submit_btn = gr.Button("🚀 提交问题", variant="primary", size="lg")
                    
                    # 答案显示区域
                    with gr.Group():
                        gr.Markdown("### 💡 系统答案")
                        answer_output = gr.Textbox(
                            label="生成的答案",
                            lines=4,
                            interactive=False
                        )
                    
                    # 统计信息
                    stats_output = gr.Textbox(
                        label="📊 处理统计",
                        lines=2,
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    # 检索文档显示区域
                    gr.Markdown("### 📄 检索到的文档")
                    docs_output = gr.Dataframe(
                        headers=["文档", "详细信息"],
                        datatype=["str", "str"],
                        row_count=10,
                        col_count=(2, "fixed"),
                        interactive=False,
                        wrap=True
                    )
            
            # 示例问题
            with gr.Accordion("📋 示例问题", open=False):
                sample_questions = self.rag_system.data_loader.get_sample_questions(5, 'train')
                if not sample_questions:
                    sample_questions = [
                        "Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?",
                        "Peter Hobbs founded the company that is based in what town in Manchester?",
                        "What direction does the river that Austrolebias bellotti are found in flow?",
                        "Who is the author of the book that mentions the city where the 1998 Winter Olympics were held?",
                        "What is the relationship between the director of Jaws and the composer of Star Wars?"
                    ]
                
                gr.Examples(
                    examples=sample_questions,
                    inputs=question_input,
                    label="点击示例问题快速测试"
                )
            
            # 系统信息
            with gr.Accordion("ℹ️ 系统信息", open=False):
                gr.Markdown(f"""
                **系统配置:**
                - 🤖 生成模型: {system_info['model_name']}
                - 🔍 检索方法: {system_info['retrieval_method']} + 余弦相似度
                - 📚 知识库: {system_info['document_count']:,} 个文档 (HotpotQA子集)
                - 🏋️ 训练样本: {system_info['train_samples']} 个
                - 📊 验证样本: {system_info['validation_samples']} 个
                - 📄 默认检索: 10个最相关文档
                
                **项目要求:**
                - COMP5423 自然语言处理 - RAG系统项目
                - 支持多跳推理问题
                - 基于检索的答案生成
                """)
            
            # 绑定事件
            submit_btn.click(
                fn=self.rag_system.rag_interface,
                inputs=[question_input, top_k_slider],
                outputs=[answer_output, stats_output, docs_output]
            )
            
            # 回车提交
            question_input.submit(
                fn=self.rag_system.rag_interface,
                inputs=[question_input, top_k_slider],
                outputs=[answer_output, stats_output, docs_output]
            )
        
        return demo
    
    def launch(self, share: bool = True, debug: bool = False):
        """启动界面"""
        print("✅ 界面构建完成！")
        print("🌐 启动Web服务...")
        
        try:
            self.demo.launch(share=share, debug=debug)
        except Exception as e:
            print(f"启动错误: {e}")
            print("尝试本地启动...")
            self.demo.launch(debug=True)

def main():
    """主函数"""
    # 安装Gradio（如果在Colab中）
    try:
        import gradio
    except ImportError:
        print("安装Gradio...")
        os.system('pip install gradio')
        import gradio as gr
    
    # 创建并启动界面
    interface = GradioInterface()
    interface.launch()

if __name__ == "__main__":
    main()
