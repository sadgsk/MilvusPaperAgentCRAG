import os
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pymilvus import connections, utility, Collection, DataType
from celery.result import AsyncResult

# 1. 结构化日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("venom_gateway")

load_dotenv(override=True)
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./venom_data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 动态导入你的核心模块 (确保与你的文件名一致)
from Parser import VenomDocumentsParser
from CeleryWorker import celery_app, task_ingest_documents_chunk
from CragFlow import crag_agent as agent_app
from services.constants import (
    FIELD_DENSE_VECTOR, FIELD_SPARSE_VECTOR,
    FIELD_CHUNK_ID, FIELD_DOI, FIELD_YEAR, FIELD_FIELD,
    get_collection_name,
)

pdf_parser=VenomDocumentsParser()

def _ensure_collection_indexes():
    col_name = get_collection_name()
    if not utility.has_collection(col_name):
        logger.info(f"Collection '{col_name}' does not exist yet; indexes will be created on first ingestion.")
        return
    collection = Collection(col_name)
    existing = {idx.field_name for idx in collection.indexes}

    if FIELD_DENSE_VECTOR not in existing:
        logger.info(f"Creating missing HNSW index on {FIELD_DENSE_VECTOR}...")
        collection.create_index(field_name=FIELD_DENSE_VECTOR,
            index_params={'index_type': 'HNSW', 'metric_type': 'IP', 'params': {'M': 16, 'efConstruction': 128}})

    if FIELD_SPARSE_VECTOR not in existing:
        logger.info(f"Creating missing SPARSE_INVERTED_INDEX on {FIELD_SPARSE_VECTOR}...")
        collection.create_index(field_name=FIELD_SPARSE_VECTOR,
            index_params={'index_type': 'SPARSE_INVERTED_INDEX', 'metric_type': 'IP', 'params': {'drop_ratio_build': 0.2}})

    for sf, idx_name in [(FIELD_CHUNK_ID, "idx_chunk_id"), (FIELD_DOI, "idx_doi"),
                         (FIELD_YEAR, "idx_year"), (FIELD_FIELD, "idx_field")]:
        if sf not in existing:
            logger.info(f"Creating missing scalar index on {sf}...")
            collection.create_index(field_name=sf, index_name=idx_name)

    logger.info("All indexes verified; pre-loading collection into memory...")
    collection.load()
    logger.info("Collection loaded and ready for queries.")

# 2. 严谨的生命周期管理 (管理 Milvus 连接池)
@asynccontextmanager
async def lifespan(app:FastAPI):
    host = os.getenv("MILVUS_HOST",'127.0.0.1')
    port = os.getenv("MILVUS_PORT", '19530')
    logger.info(f"Initializing connection to Milvus at {host}:{port}")
    try:
        connections.connect(
            alias='default',
            host=host,
            port=port
        )
        logger.info("Milvus connection established.")
        _ensure_collection_indexes()
    except Exception as e:
        logger.critical(f"FATAL: Could not connect to Milvus. {e}")
    yield
    logger.info("Closing Milvus connection.")
    connections.disconnect(alias='default')

app=FastAPI(title='Venom Scientific Agent API',version='1.0.0',lifespan=lifespan)

# 3. 生产级 CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. 全局异常捕获
@app.exception_handler(Exception)
async def global_exception_handler(request:Request,exc:Exception):
    logger.error(f"Unhandled error at {request.url.path}:{(str(exc))}")
    return JSONResponse(
        status_code=500,
        content={'message':"Internal Server Error. Please contact the administrator."}
    )

#定义模板
class ChatRequest(BaseModel):
    query:str
    session_id:str

# 5. K8s / Docker 健康检查探针
@app.get('/health')
async def health_check():
    return {'status':'healthy','service':'venom_gateway'}

# 6. 文档摄入接口 (对接解析与异步队列)
@app.post('/api/v1/ingest')
async def ingest_paper(
        doi:str = Form(...),
        year:int = Form(...),
        field:str = Form(...),
        file:UploadFile = File(...)
):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=415, detail='Only PDF files are supported.')

    if not doi or not doi.strip():
        raise HTTPException(status_code=422, detail="DOI is required.")

    max_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    file_bytes = await file.read()
    if len(file_bytes) > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {max_mb} MB.")

    file_path=os.path.join(UPLOAD_DIR,file.filename)
    try:
        with open (file_path,'wb') as f:
            f.write(file_bytes)

        logger.info(f"Received PDF {file.filename}, starting layout analysis...")

        # 异步工作，用parse.pdf将pdf转换为markdown且分块
        chunks = await asyncio.to_thread(pdf_parser.parse_pdf,file_path,doi,field,year)
        # 将分好的块传入milvus
        async_result = task_ingest_documents_chunk.delay(chunks)

        return {
            "status": "queued",
            "job_id": async_result.id,
            "message": f"File '{file.filename}' submitted to async pipeline.",
        }

    except Exception as e:
        import traceback
        logger.error(f"Ingestion failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500,detail=f"Failed to process document: {str(e)}")

@app.get("/api/v1/ingest/{job_id}")
async def ingest_status(job_id: str):
    res = AsyncResult(job_id, app=celery_app)
    payload = {"job_id": job_id, "state": res.state}

    if res.state == "SUCCESS":
        payload["result"] = res.result
    elif res.state == "FAILURE":
        payload["error"] = str(res.result)
    return payload

# 7. CRAG 大脑聊天流式接口 (SSE: Server-Sent Events)
@app.post('/api/v1/chat')
async def chat_endpoint(request:ChatRequest):
    def sse(event: str, data: dict) -> str:
        return f"event: {event}\n" f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def generate_chat_stream():
        inputs = {'question':request.query}
        config = {'configurable':{'thread_id':request.session_id}}

        try:
            last_ping = asyncio.get_running_loop().time()
            in_thinking = False
            thinking_notified = False
            buf = ""

            stream = agent_app.astream(inputs, config=config, stream_mode=["messages", "updates"])
            async for item in stream:
                now = asyncio.get_running_loop().time()
                if now - last_ping > 10:
                    last_ping = now
                    yield sse("ping", {"ts": now})

                if isinstance(item, tuple) and len(item) == 2:
                    mode, data = item

                    if mode == "messages" and isinstance(data, tuple) and len(data) == 2:
                        msg_chunk, meta = data
                        content = getattr(msg_chunk, "content", None)
                        if not content:
                            continue

                        buf += content

                        while True:
                            if in_thinking:
                                end_idx = buf.find("</think>")
                                if end_idx == -1:
                                    buf = ""
                                    break
                                buf = buf[end_idx + len("</think>"):]
                                in_thinking = False
                            else:
                                start_idx = buf.find("<think>")
                                if start_idx == -1:
                                    if buf:
                                        yield sse("token", {"content": buf, "meta": meta})
                                    buf = ""
                                    break
                                before = buf[:start_idx]
                                if before:
                                    yield sse("token", {"content": before, "meta": meta})
                                buf = buf[start_idx + len("<think>"):]
                                in_thinking = True
                                if not thinking_notified:
                                    yield sse("update", {"node": "thinking"})
                                    thinking_notified = True

                    elif mode == "updates" and isinstance(data, dict):
                        for node_name in data.keys():
                            yield sse("update", {"node": node_name})

                    continue

            yield sse("done", {})

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield sse("error", {"content": str(e)})

    return StreamingResponse(generate_chat_stream(), media_type="text/event-stream")

