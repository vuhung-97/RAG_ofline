from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import config

class TextSplitterService:
    """SRP: Chia nhỏ đoạn văn bản thô thành các chunks có metadata."""

    def __init__(self, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Phân đoạn text và gán metadata cho từng chunk (fallback cho txt/pdf)."""
        chunks = self.splitter.split_text(text)
        documents = []

        for idx, chunk in enumerate(chunks):
            doc_metadata = metadata.copy()
            doc_metadata["chunk_index"] = idx
            documents.append({
                "text": chunk,
                "metadata": doc_metadata
            })

        return documents

    def _count_tokens(self, text: str) -> int:
        # Cách đơn giản theo yêu cầu: 1 token ≈ 4 ký tự
        return max(1, len(text) // 4)

    def split_structured(self, elements: List[Dict[str, Any]], base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chia theo token 400 + overlap 100 linh động, giữ nguyên đoạn, không trộn bảng/đoạn."""
        table_max = getattr(config, "TABLE_MAX", 1000)
        chunk_tokens = getattr(config, "CHUNK_TOKENS", 400)
        overlap_tokens = getattr(config, "OVERLAP_TOKENS", 100)
        chunk_max = getattr(config, "CHUNK_TOKENS_MAX", 500)

        documents = []
        chunk_idx = 0

        # Tách riêng paragraph và table, nhưng giữ thứ tự để không trộn
        # Gom paragraph liên tiếp thành window token, bảng xử lý riêng
        i = 0
        while i < len(elements):
            el = elements[i]
            el_type = el.get("type", "paragraph")

            if el_type == "table":
                # Bảng: không gộp với đoạn, 1 bảng = 1 chunk (max 800), cắt theo dòng và lặp header
                text = el.get("text", "").strip()
                title = el.get("table_title", "")
                header_row = el.get("header_row", "")
                chapter = el.get("chapter", "")
                heading = el.get("heading", "")
                heading_level = el.get("heading_level", 0)
                if not text:
                    i += 1
                    continue
                full_text = f"{title}\n{text}" if title else text
                if self._count_tokens(full_text) <= table_max:
                    meta = base_metadata.copy()
                    meta.update({
                        "chapter": chapter, "heading": heading, "heading_level": heading_level,
                        "type": "table", "table_title": title, "header_row": header_row, "chunk_index": chunk_idx
                    })
                    documents.append({"text": full_text, "metadata": meta})
                    chunk_idx += 1
                else:
                    rows = text.split("\n")
                    # header_row là dòng đầu của text (đã có trong rows[0] nếu không tách)
                    # Nếu header_row khác rows[0], đảm bảo header được lặp
                    data_rows = rows[1:] if header_row and rows[0] == header_row else rows
                    current = ""
                    for row in data_rows:
                        # mỗi chunk con gồm title + header_row + current + row
                        prefix = ""
                        if title:
                            prefix = title + "\n"
                        if header_row:
                            prefix += header_row + "\n"
                        candidate = prefix + current + "\n" + row if current else prefix + row
                        if self._count_tokens(candidate) > table_max and current:
                            meta = base_metadata.copy()
                            meta.update({
                                "chapter": chapter, "heading": heading, "heading_level": heading_level,
                                "type": "table", "table_title": title, "header_row": header_row, "chunk_index": chunk_idx
                            })
                            chunk_text = prefix + current
                            documents.append({"text": chunk_text, "metadata": meta})
                            chunk_idx += 1
                            current = row
                        else:
                            current = candidate if not current else current + "\n" + row
                            # candidate đã có prefix, giữ nguyên
                            if current.startswith(prefix) if prefix else True:
                                pass
                    if current:
                        meta = base_metadata.copy()
                        meta.update({
                            "chapter": chapter, "heading": heading, "heading_level": heading_level,
                            "type": "table", "table_title": title, "header_row": header_row, "chunk_index": chunk_idx
                        })
                        prefix = ""
                        if title:
                            prefix = title + "\n"
                        if header_row:
                            prefix += header_row + "\n"
                        chunk_text = prefix + current if not current.startswith(prefix.strip()) else current
                        # đảm bảo prefix
                        if prefix and not chunk_text.startswith(title or header_row):
                            chunk_text = prefix + chunk_text
                        documents.append({"text": chunk_text, "metadata": meta})
                        chunk_idx += 1
                i += 1
                continue

            # Xử lý đoạn đơn vượt 500 token: cắt theo '.' fallback ';'
            if elements[i].get("type") == "paragraph":
                t_single = self._count_tokens(elements[i].get("text", ""))
                if t_single > chunk_max:
                    # đoạn đơn quá dài → cắt theo câu, mỗi phần là 1 chunk riêng
                    long_text = elements[i].get("text", "")
                    chapter = elements[i].get("chapter", "")
                    heading = elements[i].get("heading", "")
                    heading_level = elements[i].get("heading_level", 0)
                    parts = self._handle_long_paragraph_element(long_text, chunk_max)
                    for part in parts:
                        meta = base_metadata.copy()
                        meta.update({
                            "chapter": chapter, "heading": heading, "heading_level": heading_level,
                            "type": "paragraph", "table_title": "", "chunk_index": chunk_idx
                        })
                        documents.append({"text": part, "metadata": meta})
                        chunk_idx += 1
                    i += 1
                    continue

            # Gom paragraph liên tiếp cho đến ~400 token, bắt đầu đầu đoạn, kết thúc cuối đoạn
            cur_elems = []
            cur_tokens = 0
            start_i = i
            # thu thập cho tới khi đủ 400 hoặc vượt 500 thì rút
            while i < len(elements) and elements[i].get("type") == "paragraph":
                t = self._count_tokens(elements[i].get("text", ""))
                # nếu thêm sẽ vượt max 500 → rút lại 1 đoạn (không thêm)
                if cur_tokens + t > chunk_max and cur_elems:
                    break
                cur_elems.append(elements[i])
                cur_tokens += t
                i += 1
                if cur_tokens >= chunk_tokens:
                    # đã đủ 400, dừng ở cuối đoạn hiện tại
                    break

            if not cur_elems:
                i += 1
                continue

            # Tạo chunk từ cur_elems
            text_parts = [e.get("text", "") for e in cur_elems]
            chunk_text = "\n\n".join(text_parts)
            # metadata lấy từ phần tử đầu (chapter/heading)
            first = cur_elems[0]
            meta = base_metadata.copy()
            meta.update({
                "chapter": first.get("chapter", ""),
                "heading": first.get("heading", ""),
                "heading_level": first.get("heading_level", 0),
                "type": "paragraph",
                "table_title": "",
                "chunk_index": chunk_idx
            })
            documents.append({"text": chunk_text, "metadata": meta})
            chunk_idx += 1

            # Overlap ~100 token paragraph-level: lùi lại k đoạn cuối sao cho ≈100 token, không cắt giữa đoạn
            if i < len(elements) and elements[i].get("type") == "paragraph":
                overlap = 0
                k = 0
                for j in range(len(cur_elems) - 1, -1, -1):
                    overlap += self._count_tokens(cur_elems[j].get("text", ""))
                    k += 1
                    if overlap >= overlap_tokens:
                        break
                # lùi i về start + len(cur) - k, đảm bảo không cắt giữa đoạn
                if k > 0 and k < len(cur_elems):
                    i = start_i + len(cur_elems) - k

        return documents

    def _split_long_paragraph(self, text: str, max_len: int) -> List[str]:
        # cắt theo câu chỉ theo '.'; nếu không có '.' thì theo ';'
        import re
        # 1) thử cắt theo '.'
        sentences = re.split(r"(?<=\.)\s+", text)
        if len(sentences) == 1:
            # không có '.' → cắt theo ';'
            sentences = re.split(r"(?<=;)\s*", text)
        parts = []
        cur = ""
        for s in sentences:
            if len(cur) + len(s) + 1 <= max_len:
                cur = (cur + " " + s).strip()
            else:
                if cur:
                    parts.append(cur)
                # nếu câu đơn dài hơn max_len thì cắt cứng
                if len(s) > max_len:
                    for i in range(0, len(s), max_len):
                        parts.append(s[i:i+max_len])
                    cur = ""
                else:
                    cur = s
        if cur:
            parts.append(cur)
        return parts if parts else [text]

    def _handle_long_paragraph_element(self, text: str, max_tokens: int) -> List[str]:
        # Dùng cho đoạn đơn >500 token: cắt theo câu '.' fallback ';', giữ token limit
        max_len = max_tokens * 4
        parts = self._split_long_paragraph(text, max_len)
        # đảm bảo mỗi part ≤ max_tokens (ước lượng)
        filtered = []
        for p in parts:
            if self._count_tokens(p) <= max_tokens:
                filtered.append(p)
            else:
                # vẫn quá dài → cắt cứng theo max_len
                for i in range(0, len(p), max_len):
                    filtered.append(p[i:i+max_len])
        return filtered
