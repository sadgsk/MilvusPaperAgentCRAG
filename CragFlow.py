import os
import asyncio
import aiohttp
import logging
import operator
from typing import Annotated, TypedDict, List, Dict, Any
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import re

load_dotenv(override=True)
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from pymilvus import Collection, AnnSearchRequest, RRFRanker
from services.embedding import BGEEmbeddingService
from services.constants import (
    FIELD_CHUNK_ID,
    FIELD_DOI,
    FIELD_DENSE_VECTOR,
    FIELD_SPARSE_VECTOR,
    FIELD_YEAR,
    FIELD_FIELD,
    FIELD_TEXT,
    get_collection_name,
)

logger = logging.getLogger(__name__)

llm=ChatOpenAI(
    model=os.getenv('model'),
    api_key=os.getenv('api_key'),
    base_url=os.getenv("base_url"),
    temperature=0.1,
    streaming=True)
bge_service = BGEEmbeddingService()
# graphstate类其实还是个字典，所以调用的时候用[]来通过索引调用（typedict）
class GraphState(TypedDict):
    messages:Annotated[list,operator.add]
    question:str
    documents:List[Dict]
    rewritten_query:str
    evaluation_result:str
    meta_filters:str
    generation:str

async def retrieve_node(state: GraphState) -> Dict:
    logger.info("Node [retrieve]: Hybrid vector search in Milvus")

    question = state['question']
    filters = state.get('meta_filters', '')

    search_limit = int(os.getenv('RETRIEVAL_SEARCH_LIMIT', '50'))
    rrk_k = int(os.getenv('RETRIEVAL_RRF_K', '60'))
    rrk_limit = int(os.getenv('RETRIEVAL_RRF_LIMIT', '10'))

    embedding = await asyncio.to_thread(bge_service.encode_text, question)

    collection_name = get_collection_name()
    collection = Collection(collection_name)
    collection.load()

    req_dense = AnnSearchRequest(
        data=[embedding['dense']],
        anns_field=FIELD_DENSE_VECTOR,
        param={'metric_type': 'IP', 'params': {'ef': 200}},
        limit=search_limit,
        expr=filters if filters else None
    )

    req_sparse = AnnSearchRequest(
        data=[embedding['sparse']],
        anns_field=FIELD_SPARSE_VECTOR,
        param={'metric_type': 'IP', 'params': {'drop_ratio_search': 0.2}},
        limit=search_limit,
        expr=filters if filters else None
    )

    res = await asyncio.to_thread(
        collection.hybrid_search,
        reqs=[req_dense, req_sparse],
        rerank=RRFRanker(k=rrk_k),
        limit=rrk_limit,
        output_fields=[FIELD_CHUNK_ID, FIELD_DOI, FIELD_YEAR, FIELD_FIELD, FIELD_TEXT]
    )

    docs = []
    if res:
        for hit in res[0]:
            docs.append({
                'id': hit.entity.get(FIELD_CHUNK_ID),
                'doi': hit.entity.get(FIELD_DOI),
                'year': hit.entity.get(FIELD_YEAR),
                'field': hit.entity.get(FIELD_FIELD),
                'text': hit.entity.get(FIELD_TEXT),
                'distance': hit.distance
            })

    logger.info(f"Node [retrieve]: recalled {len(docs)} chunks")
    if docs:
        logger.info(f"Node [retrieve]: top hit doi={docs[0].get('doi')} score={docs[0].get('distance'):.4f}")
    return {'documents': docs}

async def evaluate_node(state: GraphState) -> Dict:
    """LLM judge: decides whether retrieved chunks sufficiently answer the question."""
    logger.info("Node [evaluate]: Judging retrieval quality")
    docs = state.get('documents', [])
    if not docs:
        logger.warning("Node [evaluate]: no chunks recalled, forcing web search fallback")
        return {"evaluation_result": "fail"}

    docs_text = '\n\n'.join([f"Snippet{i+1}:{d['text'][:1000]}" for i, d in enumerate(docs)])

    prompt = ChatPromptTemplate.from_template(
        "你是学术检索质量评估器。判断以下检索到的文献片段与用户问题是否相关。\n"
        "只要片段中包含与问题主题相关的信息，即判定为 pass；\n"
        "仅当片段内容与问题完全无关时，才判定为 fail。\n\n"
        "用户问题: {question}\n\n文献片段:\n{docs}\n\n"
        "仅回答 'pass' 或 'fail'，禁止输出其他内容。"
    )

    chain = prompt | llm
    response = await chain.ainvoke({'question': state['question'], 'docs': docs_text})

    # 拿到原始回复并转小写
    raw_result = response.content.strip().lower()
    logger.info(f"Node [evaluate]: LLM raw response = '{raw_result}'")

    # 🚨 核心修复：剔除 <think> 标签及其内部的所有思考过程
    clean_result = re.sub(r'<think>.*?</think>', '', raw_result, flags=re.DOTALL).strip()

    fail_signals = ['fail', '不相关', '无关', 'irrelevant', 'no relevant']
    # 对清洗后的干净结果进行匹配
    if any(w in clean_result for w in fail_signals):
        final_result = 'fail'
    else:
        final_result = 'pass'

    logger.info(f"Node [evaluate]: clean result matched = {final_result.upper()}")
    return {"evaluation_result": final_result}


async def rewritten_node(state: GraphState) -> Dict:
    """Rewrites the user query into concise English keywords for Semantic Scholar."""
    logger.info("Node [rewrite]: Rewriting query for academic search")
    question = state['question']
    prompt = ChatPromptTemplate.from_template(
        "将以下用户问题转换为适合在 Semantic Scholar 学术引擎中查询的极简英文关键词（不超过 5 个词）。\n"
        "用户问题：{question}\n\n"
        "仅输出关键词，不要输出任何解释性文字。"  # 稍微强化一下 prompt
    )

    chain = prompt | llm
    response = await chain.ainvoke({'question': question})

    # 核心修复：剔除 <think> 标签内部的思考过程
    raw_content = response.content.strip()
    clean_query = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

    # 去除可能包含的引号
    rewritten = clean_query.replace('"', '').replace("'", "")

    logger.info(f"Node [rewrite]: rewritten query = '{rewritten}'")
    return {'rewritten_query': rewritten}

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=16),
    reraise=True
)
async def _fetch_s2_data(query: str, api_key: str):
    """
    底层请求函数，封装了与 Semantic Scholar API 的通信及自动重试逻辑。
    """
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        headers = {}
        if api_key and api_key != "your_api_key_here":
            headers['x-api-key'] = api_key

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        # 优化：使用 params 字典，aiohttp 会自动处理空格和特殊符号的 URL 编码
        params = {
            "query": query,
            "limit": 3,
            "fields": "title,abstract,year,externalIds"
        }

        async with session.get(url, headers=headers, params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            elif resp.status == 429:
                logger.warning("Node [web_search]: 触发 429 限流，tenacity 将接管并退避重试...")
                raise Exception("Rate Limited 429")
            else:
                resp.raise_for_status()
                # 兜底异常，彻底消除 IDE 的黄线警告
                raise Exception(f"Unexpected HTTP status: {resp.status}")


async def web_search_node(state: GraphState) -> Dict:
    """Queries Semantic Scholar API as a fallback knowledge source."""
    logger.info("Node [web_search]: Querying Semantic Scholar")
    query = state.get('rewritten_query', state.get('question', ''))
    api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY', '').strip()

    try:
        data = await _fetch_s2_data(query, api_key)

        web_docs = []
        for item in data.get('data', []):
            doi = item.get('externalIds', {}).get('doi', 'Unknown')
            text = (
                f"Title: {item.get('title')}\n"
                f"Abstract: {item.get('abstract')}\n"
                f"Year: {item.get('year')}"
            )
            web_docs.append({
                # 优化：增加防御性获取，防止缺失 paperId
                'id': f"web_{item.get('paperId', 'unknown')}",
                'doi': doi,
                'text': text,
                'source': "Semantic Scholar"
            })

        current_docs = state.get('documents', [])
        current_docs.extend(web_docs)
        logger.info(f"Node [web_search]: fetched {len(web_docs)} external abstracts")
        return {'documents': current_docs}

    except Exception as e:
        logger.error(f"Node [web_search]: external search failed after retries: {e}")
        return {'documents': state.get('documents', [])}

async def generate_node(state: GraphState) -> Dict:
    """Generates the final cited answer, streaming tokens through LangGraph."""
    logger.info("Node [generate]: Generating final answer")
    docs = state['documents']

    context_str = ""
    for idx, doc in enumerate(docs):
        doi_info = f"(DOI:{doc.get('doi')})" if doc.get('doi') != 'Unknown' else ""
        context_str += f"[Source{idx+1}]{doi_info}:\n{doc['text']}\n\n"

    prompt = ChatPromptTemplate.from_template(
        "作为资深架构师和科研助手，请基于以下提供的学术文献资料回答用户问题。\n\n"
        "【绝对规则】\n"
        "1. 所有的主张必须基于提供的文献，严禁捏造（幻觉）。\n"
        "2. 必须在引用信息的句末严格标明引文来源，格式为 [Source X]。\n"
        "3. 若文献中包含数学公式、变量，请使用标准的 LaTeX 格式。\n\n"
        "文献资料:\n{context}\n\n"
        "用户问题: {question}\n\n回答:"
    )

    chain = prompt | llm
    response_content = ""
    async for chunk in chain.astream({'context': context_str, 'question': state['question']}):
        response_content += chunk.content

    import re
    response_content = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()

    return {'generation': response_content}

def edge_evaluate_node(state: GraphState) -> str:
    if state["evaluation_result"] == "pass":
        return 'generate'
    return 'rewrite'


def build_production_crag() -> StateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node('retrieve', retrieve_node)
    workflow.add_node('evaluate', evaluate_node)
    workflow.add_node('rewrite', rewritten_node)
    workflow.add_node('web_search', web_search_node)
    workflow.add_node('generate', generate_node)

    workflow.set_entry_point('retrieve')
    workflow.add_edge('retrieve', 'evaluate')
    workflow.add_conditional_edges(
        'evaluate',
        edge_evaluate_node,
        {'generate': 'generate', 'rewrite': 'rewrite'}
    )
    workflow.add_edge('rewrite', 'web_search')
    workflow.add_edge('web_search', 'generate')
    workflow.add_edge('generate', END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


crag_agent = build_production_crag()












