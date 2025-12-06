# ===============================================
#  Privacy Act 1988 - High-Precision Chunker
# ===============================================

from __future__ import annotations
import re
from src.rag.logger import get_logger
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import fitz
import tiktoken

logging = get_logger(__name__)
# --------------------------
# REGEX DEFINITIONS
# --------------------------

# Schedule header
SCHEDULE_RE = re.compile(r'^\s*Schedule\s+(\d+)\s*[—\-—]?\s*(.*)$', flags=re.IGNORECASE)

# Part header
PART_RE = re.compile(r'^\s*Part\s+([IVXLCDM]+)\s*[—\-—]?\s*(.*)$', flags=re.IGNORECASE)

# Section header (numeric or alphanumeric)
SECTION_RE = re.compile(r'^\s*((?:[1-9][0-9]?|100)[A-Z]*)\s+([A-Z][^\n]+)$')

# APP headings ("Australian Privacy Principle 4 — xxxx")
APP_RE = re.compile(
    r'^\s*(\d+)\s+Australian Privacy Principle\s+(\d+)[\s—\-—]*(.*)$',
    flags=re.IGNORECASE
)

# Schedule 2 Clause headers
CLAUSE_RE = re.compile(r'^\s*(\d+[A-Z]?)\s+(.+?)$')

# Subsections & subclauses: (1), (2A)
SUBSECTION_RE = re.compile(r'^\s*\((\d+[A-Z]?)\)\s+')

# Paragraph-level markers: (a)
PARA_RE = re.compile(r'^\s*\(([a-z])\)\s+')

# Header/footer noise patterns
HEADER_NOISE = [
    re.compile(r'^Privacy Act 1988', flags=re.IGNORECASE),
    re.compile(r'^Authorised Version', flags=re.IGNORECASE),
    re.compile(r'^Compilation No\.', flags=re.IGNORECASE),
    re.compile(r'^Page\s*\d+$', flags=re.IGNORECASE),
    re.compile(r'^\-+$'),
]

# ===============================================
# Chunker
# ===============================================

class DocumentChunker:
    def __init__(self, model_name="gpt-4o-mini", max_tokens=1000, overlap=150):
        self.max_tokens = max_tokens
        self.overlap = overlap

        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")


    # --------------------------
    # Token count helper
    # --------------------------
    def _tokens(self, text: str) -> int:
        try:
            return len(self.encoding.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    # --------------------------
    # Extract text from PDF using PyMuPDF
    # --------------------------
    def _extract_pages(self, pdf_path: str) -> List[Tuple[int, str]]:
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

                if re.match(r'^\d+$', s):  # remove page numbers
                    continue

                clean_lines.append(line.rstrip())

            if 19 <= i <= 420:  # core content
                pages.append((i, "\n".join(clean_lines)))

        return pages

    # --------------------------
    # Component segmentation
    # --------------------------
    def _segment_components(self, pages) -> List[Dict]:
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

        start("preamble", page=pages[0][0])

        for page_num, text in pages:
            detected = False

            for line in text.splitlines():
                s = line.strip()

                # Schedule
                m = SCHEDULE_RE.match(s)
                if m:
                    schedule_id = f"Schedule {m.group(1)}"
                    start("schedule", schedule_id=schedule_id,
                          title_hint=m.group(2), page=page_num - 18)
                    detected = True
                    break

                # Part
                m = PART_RE.match(s)
                if m:
                    start("main_act", page=page_num - 18, title_hint=m.group(2))
                    detected = True
                    break

                # -------- FIX #3: Detect start of Section 1 to end preamble --------
                m = SECTION_RE.match(s)
                if m and m.group(1) == "1" and current["component"] == "preamble":
                    start("main_act", page=page_num - 18, title_hint=m.group(2))
                    detected = True
                    break
                # -------------------------------------------------------------------

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

    # ---------------------------------------
    # Detect unit headings inside a component
    # ---------------------------------------
    def _detect_units(self, component_text: str, component_type: str) -> List[Dict]:
        lines = component_text.splitlines()
        headers = []

        def clean_title(t: str) -> str:
            return t.strip().rstrip(" .•-–—")

        def is_heading_line(s: str) -> bool:
            return len(s.split()) <= 15

        for i, line in enumerate(lines):
            s = line.strip()
            if not s:
                continue

            # Schedule 1 – APP
            m = APP_RE.match(s)
            if m:
                app_num = m.group(1) #or m.group(3)
                title = clean_title(m.group(3))
                headers.append((i, f"APP {app_num}", title))
                continue

            # Main Act – Section
            if component_type == "main_act":
                m = SECTION_RE.match(s)
                if m and is_heading_line(m.group(2)):
                    headers.append((i, f"section {m.group(1)}", clean_title(m.group(2))))
                    continue

            # Schedule 2 – Clause
            if component_type == "schedule":
                m = CLAUSE_RE.match(s)
                if m and is_heading_line(m.group(2)):
                    headers.append((i, f"clause {m.group(1)}", clean_title(m.group(2))))
                    continue

        if not headers:
            return [{
                "unit_id": "UNKNOWN",
                "unit_title": None,
                "start_line": 0,
                "end_line": len(lines),
                "text": component_text
            }]

        units = []
        for idx, (line_idx, unit_id, title) in enumerate(headers):
            start = line_idx
            end = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
            segment = "\n".join(lines[start:end])

            units.append({
                "unit_id": unit_id,
                "unit_title": title,
                "start_line": start,
                "end_line": end,
                "text": segment
            })

        return units

    # --------------------------
    # FIX #2 — Assign fallback IDs
    # --------------------------

    def _assign_fallback_ids(self, units):
        """
        Assigns fallback unit IDs and titles to units with 'UNKNOWN' IDs.
        The fallback ID is based on the last known unit ID, appended with '_continued'.
        """
        last_id = None
        last_title = None
        all_units = []

        for unit in units:
            if unit["metadata"]["unit_id"] == "UNKNOWN":
                unit["metadata"]["unit_id"] = f"{last_id}_continued" if last_id else None
                unit["metadata"]["unit_title"] = f"{last_title} (continued)" if last_title else None
                
            else:
                last_id = unit["metadata"]["unit_id"]
                last_title = unit["metadata"]["unit_title"]
            all_units.append(unit)
        return all_units[1:]

    # --------------------------
    # Chunk splitting logic
    # --------------------------
    def _split_into_chunks(self, unit: Dict, meta: Dict) -> List[Dict]:
        text = unit["text"]

        # -------- FIX #1: remove text & line numbers from metadata --------
        unit_meta = {
            k: v for k, v in unit.items()
            if k not in ("text", "start_line", "end_line")
        }
        # ------------------------------------------------------------------

        if self._tokens(text) <= self.max_tokens:
            return [{
                "text": text,
                "token_count": self._tokens(text),
                "metadata": {**meta, **unit_meta}
            }]

        parts = re.split(r'(?=\n\s*\(\d+[A-Z]?\)\s+)', text)
        final = []

        for part in parts:
            if not part.strip():
                continue
            if self._tokens(part) <= self.max_tokens:
                final.append(part)
            else:
                paras = re.split(r'(?=\n\s*\([a-z]\)\s+)', part)
                for para in paras:
                    if not para.strip():
                        continue
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
                        if buf:
                            final.append(buf.strip())

        chunks = []
        for f in final:
            chunks.append({
                "text": f.strip(),
                "token_count": self._tokens(f),
                "metadata": {**meta, **unit_meta}
            })

        return chunks

    # --------------------------
    # PUBLIC API
    # --------------------------
    def create_chunks(self, pdf_path: str) -> List[Dict]:
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

        all_chunks = self._assign_fallback_ids(all_chunks)
        return all_chunks
