================【请将以下内容保存为 requirements.txt】================
fastapi
uvicorn[standard]
streamlit
requests
langchain
langchain-openai
langchain-text-splitters
langgraph
pymilvus
FlagEmbedding
celery
redis
aiohttp
python-dotenv
pydantic
magic-pdf[full]


================【请将以下内容保存为 README.md】================
# 🐍 Venom: Scientific Paper CRAG Agent

Venom 是一个面向学术论文的**纠正式检索增强生成（CRAG）Agent**。支持将双栏学术 PDF 注入本地 Milvus 向量数据库，进行带引用的精准问答。当本地知识库不足时，系统会自动 fallback 到 Semantic Scholar 进行学术检索。
!项目演示图](img.png)
## 🚀 快速启动指南

### 1. 环境准备
推荐使用 Python 3.10 或 3.11。克隆项目后，创建并激活虚拟环境：

```bash
git clone <your-repo-url>
cd MelivusCrag

python -m venv .venv
# Windows 激活
.venv\Scripts\Activate.ps1
# macOS/Linux 激活
source .venv/bin/activate
```

安装核心依赖：
```bash
pip install -r requirements.txt
```

### 2. 启动基础架构 (Docker)
项目依赖 Milvus (>=2.4.5) 和 Redis。在项目根目录下运行：
```bash
docker compose up -d
```
使用 `docker compose ps` 确保 etcd, minio, milvus, redis 均处于 healthy/running 状态。

### 3. 环境配置
复制环境变量模板：
```bash
# Windows
copy .env.example .env  
# macOS/Linux
cp .env.example .env
```
在 `.env` 中填入你的 API 密钥：
```ini
deepseek_api_key = "sk-xxxxxxxxxxxxxxxx"
SEMANTIC_SCHOLAR_API_KEY = "your_key_here" # 选填，用于外网学术搜索
```

### 4. PDF 解析模型配置 (MinerU)
在用户主目录（如 `C:\Users\你的用户名\`）下创建 `magic-pdf.json`：
```json
{
    "models-dir": "D:\\magic-pdf-models\\models",
    "layoutreader-model-dir": "D:\\magic-pdf-models\\models\\ReadingOrder\\layout_reader",
    "device-mode": "cuda",
    "table-config": { "model": "rapid_table", "enable": false, "max_time": 400 },
    "formula-config": { "mfd_model": "yolo_v8_mfd", "mfr_model": "unimernet_small", "enable": false },
    "layout-config": { "model": "doclayout_yolo" }
}
```
*注：请确保 `models-dir` 指向你本地的 PDF-Extract-Kit 模型路径。若无 GPU，将 `device-mode` 改为 `cpu`。*

### 5. 启动系统进程
请在项目根目录打开**三个**独立的终端（确保都激活了虚拟环境），依次启动：

**终端 1：向量入库队列 (Celery)**
```bash
celery -A CeleryWorker worker --loglevel=info --pool=solo
```
*(注：Windows 系统必须加 `--pool=solo` 参数)*

**终端 2：核心大模型 API (FastAPI)**
```bash
python -c "import uvicorn; uvicorn.run('main:app', host='0.0.0.0', port=8000)"
```

**终端 3：交互界面 (Streamlit)**
```bash
streamlit run app.py
```

启动完成后，浏览器会自动打开 `http://localhost:8501`。在左侧边栏上传 PDF 注入知识库，随后即可在主界面进行对话问答！