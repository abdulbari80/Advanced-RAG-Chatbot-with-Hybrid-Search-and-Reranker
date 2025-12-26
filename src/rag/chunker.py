# ===============================================
#  Privacy Act 1988 - High-Precision Chunker
# ===============================================

from __future__ import annotations
import re
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import fitz
import tiktoken

from src.rag.logger import get_logger
from src.rag.exception import DocumentLoadError, RAGBaseException

logging = get_logger(__name__)

# --------------------------
# REGEX DEFINITIONS (Structural Markers)
# --------------------------
SCHEDULE_RE = re.compile(r'^\s*Schedule\s+(\d+)\s*[—\-—]?\s*(.*)$', flags=re.IGNORECASE)
PART_RE = re.compile(r'^\s*Part\s+([IVXLCDM]+)\s*[—\-—]?\s*(.*)$', flags=re.IGNORECASE)
SECTION_RE = re.compile(r'^\s*((?:[1-9][0-9]?|100)[A-Z]*)\s+([A-Z][^\n]+)$')
APP_RE = re.compile(r'^\s*(\d+)\s+Australian Privacy Principle\s+(\d+)[\s—\-—]*(.*)$', flags=re.IGNORECASE)
CLAUSE_RE = re.compile(r'^\s*(\d+[A-Z]?)\s+(.+?)$')
SUBSECTION_RE = re.compile(r'^\s*\((\d+[A-Z]?)\)\s+')
PARA_RE = re.compile(r'^\s*\(([a-z])\)\s+')

# Header/footer noise patterns for document cleaning
HEADER_NOISE = [
    re.compile(r'^Privacy Act 1988', flags=re.IGNORECASE),
    re.compile(r'^Authorised Version', flags=re.IGNORECASE),
    re.compile(r'^Compilation No\.', flags=re.IGNORECASE),
    re.compile(r'^Page\s*\d+$', flags=re.IGNORECASE),
    re.compile(r'^\-+$'),
]

class DocumentChunker:
    """
    A specialized document processor for the Privacy Act 1988.
    
    This class handles the end-to-end ingestion pipeline from raw PDF extraction 
    to hierarchical structural segmentation and final token-based chunking.
    
    Attributes:
        max_tokens (int): Maximum allowable tokens per chunk.
        overlap (int): Token overlap between chunks (if applicable).
        encoding (tiktoken.Encoding): Tokenizer specific to the target LLM.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", max_tokens: int = 1000, overlap: int = 150):
        """
        Initializes the chunker with model-specific tokenization settings.
        """
        self.max_tokens = max_tokens
        self.overlap = overlap

        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except Exception as e:
            logging.warning(f"Failed to load encoding for {model_name}, falling back to cl100k_base. Error: {e}")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def _tokens(self, text: str) -> int:
        """
        Calculates the token count for a given string.

        Args:
            text (str): The string to tokenize.

        Returns:
            int: Number of tokens.
        """
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback: Approximate 4 characters per token
            return max(1, len(text) // 4)

    def _extract_pages(self, pdf_path: str) -> List[Tuple[int, str]]:
        """
        Extracts and cleans raw text from PDF pages using PyMuPDF.

        Args:
            pdf_path (str): Path to the source PDF file.

        Returns:
            List[Tuple[int, str]]: A list of (page_number, text) tuples.

        Raises:
            DocumentLoadError: If the PDF cannot be opened or processed.
        """
        try:
            doc = fitz.open(pdf_path)
            pages = []

            for i, p in enumerate(doc, start=1):
                raw = p.get_text("text") or ""
                clean_lines = []

                for line in raw.splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    if any(rx.match(s) for rx in HEADER_NOISE):
                        continue
                    if re.match(r'^\d+$', s):  # remove standalone page numbers
                        continue
                    clean_lines.append(line.rstrip())

                # Filter for core content pages (Specific to Privacy Act PDF layout)
                if 19 <= i <= 420:
                    pages.append((i, "\n".join(clean_lines)))
            
            doc.close()
            return pages
        except Exception as e:
            logging.error(f"Error extracting PDF pages from {pdf_path}: {str(e)}")
            raise DocumentLoadError(f"PyMuPDF failed to process {pdf_path}. Details: {e}")

    def _segment_components(self, pages: List[Tuple[int, str]]) -> List[Dict]:
        """
        Groups pages into major legislative components (Preamble, Main Act, Schedules).

        Args:
            pages (List[Tuple[int, str]]): Extracted page content.

        Returns:
            List[Dict]: High-level document components with associated text and metadata.
        """
        units = []
        current = None

        def start(component: str, schedule_id=None, title_hint=None, page=0):
            nonlocal current
            if current:
                units.append(current)
            current = {
                "component": component,
                "schedule_id": schedule_id,
                "start_page": page,
                "end_page": page,
                "title_hint": title_hint,
                "lines": []
            }

        try:
            start("preamble", page=pages[0][0])

            for page_num, text in pages:
                detected = False
                for line in text.splitlines():
                    s = line.strip()

                    m_sched = SCHEDULE_RE.match(s)
                    if m_sched:
                        schedule_id = f"Schedule {m_sched.group(1)}"
                        start("schedule", schedule_id=schedule_id,
                              title_hint=m_sched.group(2), page=page_num - 18)
                        detected = True
                        break

                    m_part = PART_RE.match(s)
                    if m_part:
                        start("main_act", page=page_num - 18, title_hint=m_part.group(2))
                        detected = True
                        break

                    m_sect = SECTION_RE.match(s)
                    if m_sect and m_sect.group(1) == "1" and current["component"] == "preamble":
                        start("main_act", page=page_num - 18, title_hint=m_sect.group(2))
                        detected = True
                        break

                if current is None:
                    start("main_act", page=page_num - 18)

                current["lines"].append(text)
                current["end_page"] = page_num - 18

            if current:
                units.append(current)

            for unit in units:
                unit["text"] = "\n".join(unit["lines"]).strip()
                del unit["lines"]

            return units
        except Exception as e:
            logging.error(f"Error during component segmentation: {e}")
            raise RAGBaseException(f"Failed to segment document into components: {e}")

    def _detect_units(self, component_text: str, component_type: str) -> List[Dict]:
        """
        Identifies granular units (Sections, Clauses, APPs) within a component.

        Args:
            component_text (str): The full text of the component.
            component_type (str): Type of component (main_act, schedule, etc.).

        Returns:
            List[Dict]: List of units with IDs and their respective text segments.
        """
        lines = component_text.splitlines()
        headers = []

        def clean_title(t: str) -> str:
            return t.strip().rstrip(" .•-–—")

        def is_heading_line(s: str) -> bool:
            return len(s.split()) <= 15

        for i, line in enumerate(lines):
            s = line.strip()
            if not s: continue

            m_app = APP_RE.match(s)
            if m_app:
                headers.append((i, f"APP {m_app.group(1)}", clean_title(m_app.group(3))))
                continue

            if component_type == "main_act":
                m_sect = SECTION_RE.match(s)
                if m_sect and is_heading_line(m_sect.group(2)):
                    headers.append((i, f"section {m_sect.group(1)}", clean_title(m_sect.group(2))))
                    continue

            if component_type == "schedule":
                m_cl = CLAUSE_RE.match(s)
                if m_cl and is_heading_line(m_cl.group(2)):
                    headers.append((i, f"clause {m_cl.group(1)}", clean_title(m_cl.group(2))))
                    continue

        if not headers:
            return [{
                "unit_id": "UNKNOWN", "unit_title": None,
                "start_line": 0, "end_line": len(lines), "text": component_text
            }]

        units = []
        for idx, (line_idx, unit_id, title) in enumerate(headers):
            start_idx = line_idx
            end_idx = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
            segment = "\n".join(lines[start_idx:end_idx])

            units.append({
                "unit_id": unit_id, "unit_title": title,
                "start_line": start_idx, "end_line": end_idx, "text": segment
            })
        return units

    def _assign_fallback_ids(self, chunks: List[Dict]) -> List[Dict]:
        """
        Propagates context to chunks labeled as 'UNKNOWN' using forward-fill logic.

        Args:
            chunks (List[Dict]): Chunks generated from the splitting process.

        Returns:
            List[Dict]: Chunks with enriched metadata and fallback IDs.
        """
        last_id = None
        last_title = None
        all_units = []

        for unit in chunks:
            meta = unit["metadata"]
            if meta.get("unit_id") == "UNKNOWN":
                meta["unit_id"] = f"{last_id}_continued" if last_id else None
                meta["unit_title"] = f"{last_title} (continued)" if last_title else None
            else:
                last_id = meta.get("unit_id")
                last_title = meta.get("unit_title")
            all_units.append(unit)
        
        # Note: Added skip for initial artifact if necessary, keeping your slice logic
        return all_units[1:] if len(all_units) > 1 else all_units

    def _split_into_chunks(self, unit: Dict, meta: Dict) -> List[Dict]:
        """
        Splits a legislative unit into smaller chunks using a hierarchical regex strategy.
        
        Tries splitting by Subsections first, then Paragraphs, then Sentences to preserve
        legal formatting context.

        Args:
            unit (Dict): The unit to split.
            meta (Dict): Base metadata for the component.

        Returns:
            List[Dict]: List of chunks with text and flattened metadata.
        """
        text = unit["text"]
        unit_meta = {k: v for k, v in unit.items() if k not in ("text", "start_line", "end_line")}

        if self._tokens(text) <= self.max_tokens:
            return [{"text": text, "token_count": self._tokens(text), "metadata": {**meta, **unit_meta}}]

        # Hierarchical split: (Subsections) -> (Paragraphs) -> (Sentences)
        parts = re.split(r'(?=\n\s*\(\d+[A-Z]?\)\s+)', text)
        final = []

        for part in parts:
            if not part.strip(): continue
            if self._tokens(part) <= self.max_tokens:
                final.append(part)
            else:
                paras = re.split(r'(?=\n\s*\([a-z]\)\s+)', part)
                for para in paras:
                    if not para.strip(): continue
                    if self._tokens(para) <= self.max_tokens:
                        final.append(para)
                    else:
                        sentences = re.split(r'(?<=[.!?])\s+', para)
                        buf = ""
                        for s in sentences:
                            if self._tokens(buf + " " + s) > self.max_tokens:
                                final.append(buf.strip())
                                buf = s
                            else:
                                buf += " " + s
                        if buf: final.append(buf.strip())

        return [{
            "text": f.strip(),
            "token_count": self._tokens(f),
            "metadata": {**meta, **unit_meta}
        } for f in final]

    def create_chunks(self, pdf_path: str) -> List[Dict]:
        """
        Public API: Orchestrates the transformation from PDF file to vectorized chunks.

        Args:
            pdf_path (str): File path to the Privacy Act PDF.

        Returns:
            List[Dict]: A flattened list of chunks ready for embedding.
            
        Raises:
            DocumentLoadError: If PDF extraction fails.
            RAGBaseException: For general pipeline processing errors.
        """
        try:
            pages = self._extract_pages(pdf_path)
            components = self._segment_components(pages)

            all_chunks = []
            idx = 0

            for comp in components:
                base_meta = {
                    "source": "Privacy Act 1988",
                    "component": comp["component"],
                    "schedule": comp["schedule_id"],
                    "page_range": f"{comp['start_page']}-{comp['end_page']}"
                }

                units = self._detect_units(comp["text"], comp["component"])

                for unit in units:
                    chunks = self._split_into_chunks(unit, base_meta)
                    for chunk in chunks:
                        chunk["metadata"]["chunk_index"] = idx
                        idx += 1
                        all_chunks.append(chunk)

            return self._assign_fallback_ids(all_chunks)
        except (DocumentLoadError, RAGBaseException) as e:
            # Re-raise known exceptions to be handled by the caller
            raise e
        except Exception as e:
            logging.critical(f"Unexpected failure in create_chunks: {e}")
            raise RAGBaseException(f"An unexpected error occurred during ingestion: {e}")