import os
import tempfile
import logging
import shutil
from typing import List, Dict, Any
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import FileBasedDataWriter


# 引入工业级的文本分割器
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger=logging.getLogger(__name__)

class VenomDocumentsParser:
    def __init__(self):


        headers_to_split_on=[
            ('#','Header1'),
            ('##','Header2'),
            ('###','Header3')
        ]

        self.md_spliter=MarkdownHeaderTextSplitter( headers_to_split_on= headers_to_split_on)

        self.char_spliter=RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            # 优先级：先按双换行切(段落)，再按单换行切，尽量不破坏内部结构
            separators=["\n\n", "\n", " ", ""]
        )

    def parse_pdf(self,pdf_path:str,doi:str,field:str,year:int) -> List[Dict[str, Any]]:
        base_tmp=tempfile.gettempdir()
        local_image_dir=os.path.join(base_tmp,'venom_images',doi.replace('/','_'))
        os.makedirs(local_image_dir,exist_ok=True)

        try:
            image_dir=os.path.basename(local_image_dir)
            image_writer=FileBasedDataWriter(local_image_dir)

            with open (pdf_path,'rb') as f:
                pdf_bytes=f.read()
            ds=PymuDocDataset(pdf_bytes)

            method = ds.classify()
            use_ocr = False
            try:
                use_ocr = method == SupportedPdfParseMethod.OCR
            except Exception:
                use_ocr = "ocr" in str(method).lower()

            if use_ocr:
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)

            md_content=pipe_result.get_markdown(image_dir)

            chunks = self._smart_chunking(md_content, doi=doi, year=year, field=field)
            logger.info(f"Successfully parsed PDF into {len(chunks)} smart chunks.")
            return chunks

        except Exception as e:
            logger.error(f"Failed to parse PDF {pdf_path}: {e}")
            raise

        finally:
            if os.path.exists(local_image_dir):
                shutil.rmtree(local_image_dir,ignore_errors=True)

    def _smart_chunking(self,md_text:str,doi:str,year:int,field:str)-> List[Dict[str, Any]]:
        final_chunks=[]
        md_header_splits=self.md_spliter.split_text(md_text)
        chunk_index=0
        for split in md_header_splits:
            sub_splits=self.char_spliter.split_text(split.page_content)
            for sub_text in sub_splits:
                context_header = (
                    ">".join([str(v) for v in split.metadata.values() if v]) if split.metadata else ""
                )
                enriched_text = f"[{context_header}]\n{sub_text}" if context_header else sub_text

                final_chunks.append({
                    "chunk_id": f"{doi}_chunk_{chunk_index}",
                    "doi": doi,
                    "year": year,
                    "field": field,  # 我们刚才着重讨论的过滤字段
                    "text": enriched_text
                })
                chunk_index+=1

        return final_chunks










