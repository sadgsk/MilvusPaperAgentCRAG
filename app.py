import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
import uuid
import time

load_dotenv(override=True)
FASTAPI_URL=os.getenv('FASTAPI_BASE_URL',"http://127.0.0.1:8000")
st.set_page_config(page_title="Venom: 工业级科研 Agent", layout="wide", page_icon="🐍")

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

st.title("🐍 Venom: Scientific Paper Assistant")

# 1. 状态管理安全初始化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_status" not in st.session_state:
    st.session_state.system_status = "unknown"
if "last_ingest_job_id" not in st.session_state:
    st.session_state.last_ingest_job_id = None
# 2. 侧边栏：后端健康检查与文献注入控制台
with st.sidebar:
    st.header("⚙️ System Status")
    try:
        health_res = requests.get(f"{FASTAPI_URL}/health",timeout=2)
        if health_res.status_code == 200:
            st.success("✅ Backend Online")
        else:
            st.warning("⚠️ Backend Degraded")
    except requests.exceptions.RequestException:
        st.error("🚨 Backend Offline! Check Docker/FastAPI.")

    st.markdown("---")
    st.header("🗄️ 知识库注入 (Ingestion)")

    with st.form("ingest_form", clear_on_submit=True):
        upload_file = st.file_uploader("上传双栏学术 PDF",type=['pdf'])
        input_doi = st.text_input("DOI (如 10.1038/s41586-024)", placeholder="必填项")
        input_year = st.number_input("发表年份", min_value=1950, max_value=2030, value=2024)

        input_field = st.selectbox(
            "所属学科领域",
            ["Resources and Environment", "Computer Science", "Bioinformatics", "Other"]
        )

        submit_btn = st.form_submit_button("🚀 深度解析并入库")

        if submit_btn and upload_file and input_doi:
            with st.spinner("📦 正在传输至后端处理队列..."):
                try:
                    files = {"file": (upload_file.name, upload_file, "application/pdf")}
                    data = {"doi": input_doi.strip(), "year": int(input_year), "field": input_field}

                    res=requests.post(f"{FASTAPI_URL}/api/v1/ingest",files=files,data=data)
                    res.raise_for_status()
                    payload = res.json()
                    st.session_state.last_ingest_job_id = payload.get("job_id")
                    if st.session_state.last_ingest_job_id:
                        st.info(f"已提交入库任务 job_id: `{st.session_state.last_ingest_job_id}`")
                except Exception as e:
                    st.error(f"入库提交失败: {str(e)}")
        elif submit_btn:
            st.warning('请补全 PDF 文件和 DOI 信息。')

    if st.session_state.last_ingest_job_id:
        st.markdown("---")
        st.subheader("📌 入库任务状态")
        try:
            status_res = requests.get(
                f"{FASTAPI_URL}/api/v1/ingest/{st.session_state.last_ingest_job_id}",
                timeout=2,
            )
            if status_res.status_code == 200:
                st.json(status_res.json())
            else:
                st.warning(f"状态查询失败: HTTP {status_res.status_code}")
        except requests.exceptions.RequestException as e:
            st.warning(f"状态查询失败: {e}")
        if st.button("刷新状态"):
            st.rerun()
# 3. 渲染历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# 4. 核心对话交互与 SSE 数据流接收
if prompt := st.chat_input("请输入问题："):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        full_response = ""

        req_data = {
            "query": prompt,
            "session_id": st.session_state.current_session_id
        }

        try:
            with requests.post(f"{FASTAPI_URL}/api/v1/chat",json=req_data, stream=True, timeout=120) as r:
                r.raise_for_status()

                current_event = None
                for raw_line in r.iter_lines(decode_unicode=True):
                    if raw_line is None:
                        continue
                    line = raw_line.strip()
                    if not line:
                        continue

                    if line.startswith("event:"):
                        current_event = line.split(":", 1)[1].strip()
                        continue

                    if line.startswith("data:"):
                        payload = line.split(":", 1)[1].strip()
                        if not payload:
                            continue
                        try:
                            event_data = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        evt = current_event
                        if evt == "update":
                            current_node = event_data.get("node")
                            if current_node:
                                if current_node == "thinking":
                                    status_placeholder.caption("正在思考...")
                                else:
                                    status_placeholder.caption(f"LangGraph: 进入 **{current_node}** 节点...")
                        elif evt == "token":
                            piece = event_data.get("content") or ""
                            if piece:
                                full_response += piece
                                message_placeholder.markdown(full_response)
                        elif evt == "ping":
                            pass
                        elif evt == "done":
                            status_placeholder.empty()
                            message_placeholder.markdown(full_response)
                        elif evt == "error":
                            st.error(f"后端图执行报错: {event_data.get('content')}")
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except requests.exceptions.Timeout:
            st.error("❌ 请求超时。后端的 BGE-M3 或大模型响应时间过长。")
        except requests.exceptions.ConnectionError:
            st.error("❌ 无法连接到 FastAPI 后端，请检查网络或端口设置。")
        except Exception as e:
            st.error(f"❌ 通信异常: {str(e)}")




