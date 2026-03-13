import logging
from typing import Dict, Any
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)


class BGEEmbeddingService:
    """全局统一的 BGE-M3 向量化单例服务"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BGEEmbeddingService, cls).__new__(cls)
            logger.info("Loading BGE-M3 model to GPU...")
            cls._instance.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
            logger.info("BGE-M3 loaded successfully.")
        return cls._instance

    def encode_text(self, text: str) -> Dict[str, Any]:
        """统一的特征提取接口 (同时适用于短 Query 和长 Chunk)"""
        embeddings = self.model.encode(
            [text], return_dense=True, return_sparse=True, return_colbert_vecs=False
        )

        sparse_raw = embeddings['lexical_weights'][0]
        milvus_sparse = {}
        for tok, weight in sparse_raw.items():
            token_id = int(tok) if isinstance(tok, (int, str)) else tok
            if token_id >= 0:
                milvus_sparse[token_id] = float(weight)

        return {
            "dense": embeddings['dense_vecs'][0].tolist(),
            "sparse": milvus_sparse
        }


