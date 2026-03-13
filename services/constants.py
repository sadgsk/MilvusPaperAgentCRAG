import os


def get_collection_name() -> str:
    return os.getenv("VENOM_COLLECTION_NAME", "venom_papers_prod")


# Milvus field names (single source of truth)
FIELD_CHUNK_ID = "chunk_id"
FIELD_DOI = "doi"
FIELD_YEAR = "year"
FIELD_FIELD = "field"
FIELD_TEXT = "text"

FIELD_DENSE_VECTOR = "dense_vector"
FIELD_SPARSE_VECTOR = "sparse_vector"


REQUIRED_FIELDS = {
    FIELD_CHUNK_ID,
    FIELD_DOI,
    FIELD_YEAR,
    FIELD_FIELD,
    FIELD_TEXT,
    FIELD_DENSE_VECTOR,
    FIELD_SPARSE_VECTOR,
}

