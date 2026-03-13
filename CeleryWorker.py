import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from celery import Celery
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType, MilvusException
from FlagEmbedding import BGEM3FlagModel
from services.embedding import BGEEmbeddingService
from services.constants import (
    FIELD_CHUNK_ID,
    FIELD_DOI,
    FIELD_DENSE_VECTOR,
    FIELD_SPARSE_VECTOR,
    FIELD_YEAR,
    FIELD_FIELD,
    FIELD_TEXT,
    REQUIRED_FIELDS,
    get_collection_name,
)

# 配置全局日志规则：设置显示级别为 INFO，并定义时间、级别、内容的显示格式
logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
# 创建当前模块的专属记录器：自动以当前文件名命名，方便在日志中精准定位来源
logger=logging.getLogger(__name__)
# override保障env中的变量可以覆盖
load_dotenv(override=True)

MILVUS_HOST=os.getenv('MILVUS_HOST','127.0.0.1')
MILVUS_PORT=os.getenv('MILVUS_PORT','19530')
COLLECTION_NAME = get_collection_name()


# 1. 实例化 Celery App
_redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
celery_app = Celery('venom_ingestion', broker=_redis_url, backend=_redis_url)

# Milvus 管理器：将数据传入向量数据库milvus
class MilvusManager:
    #启动函数
    def __init__(self):
        self._connect()
        self.collection=self._init_collection()

    def _connect(self):
        try:
            connections.connect(
                alias='default',
                host=MILVUS_HOST,
                port=MILVUS_PORT,
                timeout=10.0
            )
        except MilvusException as e:
            logger.exception(f"Milvus connect error{str(e)}")
            raise

    def _init_collection(self) ->Collection:
        if utility.has_collection(COLLECTION_NAME):
            logger.info(f"Collection {COLLECTION_NAME} already exists.")
            collection = Collection(COLLECTION_NAME)
            existing_fields = {f.name for f in collection.schema.fields}
            missing = REQUIRED_FIELDS - existing_fields
            if missing:
                raise RuntimeError(
                    f"Milvus collection schema mismatch for '{COLLECTION_NAME}'. Missing fields: {sorted(missing)}. "
                    f"Fix by migrating/recreating collection or changing VENOM_COLLECTION_NAME."
                )
            self._ensure_indexes(collection)
            collection.load()
            return collection

        fields=[
            FieldSchema(name=FIELD_CHUNK_ID, dtype=DataType.VARCHAR, max_length=128, is_primary=True),
            FieldSchema(name=FIELD_DOI, dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name=FIELD_YEAR, dtype=DataType.INT16),
            FieldSchema(name=FIELD_FIELD, dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name=FIELD_TEXT, dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name=FIELD_DENSE_VECTOR, dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name=FIELD_SPARSE_VECTOR, dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]

        schema=CollectionSchema(fields=fields,description='Venom Scientific Papers Collection',enable_dynamic_schema=False)
        collection=Collection(name=COLLECTION_NAME,schema=schema)

        self._ensure_indexes(collection)

        logger.info("正在将集合加载到内存...")
        collection.load()

        return collection

    def _ensure_indexes(self, collection: Collection):
        existing_indexes = {idx.field_name for idx in collection.indexes}

        if FIELD_DENSE_VECTOR not in existing_indexes:
            logger.info(f"Creating HNSW index on {FIELD_DENSE_VECTOR}...")
            dense_index={'index_type':'HNSW','metric_type':'IP','params':{'M':16,'efConstruction':128}}
            collection.create_index(field_name=FIELD_DENSE_VECTOR, index_params=dense_index)

        if FIELD_SPARSE_VECTOR not in existing_indexes:
            logger.info(f"Creating SPARSE_INVERTED_INDEX on {FIELD_SPARSE_VECTOR}...")
            sparse_index={'index_type':'SPARSE_INVERTED_INDEX','metric_type':'IP','params':{'drop_ratio_build':0.2}}
            collection.create_index(field_name=FIELD_SPARSE_VECTOR, index_params=sparse_index)

        for scalar_field, idx_name in [
            (FIELD_CHUNK_ID, "idx_chunk_id"),
            (FIELD_DOI, "idx_doi"),
            (FIELD_YEAR, "idx_year"),
            (FIELD_FIELD, "idx_field"),
        ]:
            if scalar_field not in existing_indexes:
                logger.info(f"Creating scalar index on {scalar_field}...")
                collection.create_index(field_name=scalar_field, index_name=idx_name)

    def insert_batch(self,data:List[Dict[str,Any]]):
        try:
            res=self.collection.insert(data)
            self.collection.flush()
            logger.info(f'Successfully inserted and flushed {len(data)} chunks.')
            return res.primary_keys
        except Exception as e:
            logger.exception(f"Ingestion failed:{str(e)}")
            raise

bge_service = BGEEmbeddingService()
_milvus_manager: MilvusManager | None = None


def get_milvus_manager() -> MilvusManager:
    global _milvus_manager
    if _milvus_manager is None:
        _milvus_manager = MilvusManager()
    return _milvus_manager

@celery_app.task(bind=True, max_retries=3)
def task_ingest_documents_chunk(self,chunks:List[Dict[str,Any]]):
    """
    接收 parser.py 切好的纯文本 chunks -> 动用 GPU 算向量 -> 拼装 -> 批量写入 Milvus
    """
    #传进来的chunks只有id，doi等这些标量信息，这个函数就是负责将稠密向量与稀疏向量加入chunks
    try:
        enrich_chunks=[]
        for chunk in chunks:
            # 将chunk中的'text'索引对应的部分传进函数并将返回的稠密向量与稀疏向量赋给vecs
            vecs=bge_service.encode_text(chunk['text'])
            # 这里新建的索引名称必须和前面的fieldname保持一致
            chunk['dense_vector']=vecs['dense']
            chunk['sparse_vector']=vecs['sparse']

            enrich_chunks.append(chunk)

        logger.info("Vectorization complete. Pushing to Milvus...")
        primary_keys = get_milvus_manager().insert_batch(enrich_chunks)
        return {'status':'success','insert count':len(primary_keys)}

    except Exception as exc:
        logger.warning(f"Ingestion failed, retrying... ({self.request.retries}/3)")
        raise self.retry(exc=exc,countdown=5**self.request.retries)







