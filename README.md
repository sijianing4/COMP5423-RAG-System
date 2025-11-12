# COMP5423-RAG-System
# COMP5423 RAG System

基于HotpotQA数据集的检索增强生成问答系统。

## 🚀 项目简介

这是一个完整的RAG系统，包含：
- 🔍 **检索模块**: TF-IDF检索器
- 🤖 **生成模块**: 基于Qwen模型的答案生成
- 🎨 **用户界面**: Gradio Web界面
- 📊 **评估系统**: 完整的模块化架构
## 📁 项目结构
COMP5423-RAG-System/
├── retrieval/ # 检索模块
│ ├── init.py
│ └── tfidf_retriever.py # TF-IDF检索器
├── generation/ # 生成模块
│ ├── init.py
│ └── basic_generator.py # 基础生成器
├── integration/ # 集成模块
│ ├── init.py
│ ├── rag_system.py # 主RAG系统
│ └── gradio_ui.py # 用户界面
├── utils/ # 工具模块
│ ├── init.py
│ └── data_loader.py # 数据加载器
├── interface/ # 接口定义
├── tests/ # 测试代码
├── notebooks/ # Jupyter笔记本
├── main.py # 主运行文件
├── requirements.txt # 依赖包
└── README.md # 项目说明
## 🛠️ 快速开始

### 在Google Colab中运行

```python
# 克隆仓库
!git clone https://github.com/your-username/COMP5423-RAG-System.git
%cd COMP5423-RAG-System

# 安装依赖
!pip install -r requirements.txt

# 运行系统
!python main.py
#运行模式
命令行演示模式: 交互式测试问题

Web界面模式: 启动Gradio Web界面

系统测试模式: 运行系统测试
#模块说明
检索模块 (retrieval/)
TFIDFRetriever: 基于TF-IDF和余弦相似度的文档检索

生成模块 (generation/)
BasicGenerator: 基于Qwen模型的答案生成器

工具模块 (utils/)
DataLoader: 数据加载和处理

集成模块 (integration/)
RAGSystem: 主系统集成

GradioInterface: Web用户界面
#数据说明
使用HotpotQA数据集的子集：

训练集: 12,000样本

验证集: 1,500样本

文档集: 144,718文档
#团队成员
成员A: 检索模块开发

成员B: 生成模块开发

成员C: 界面和集成

成员D: 测试和文档
#许可证
MIT License
with open('README.md', 'w', encoding='utf-8') as f:
f.write(readme_content)
print("✅ 创建 README.md")

##
###🚀 使用方法

现在你可以通过以下方式运行系统：

```python
# 方式1: 运行主程序
!python main.py

# 方式2: 直接运行界面
!python integration/gradio_ui.py

# 方式3: 运行命令行演示
!python integration/rag_system.py
