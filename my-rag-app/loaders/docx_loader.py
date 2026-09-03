import re
import docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from loaders.base import BaseLoader


class DocxLoader(BaseLoader):
    """SRP: Trích xuất văn bản thô từ file Microsoft Word (.docx) - heading-aware."""

    def load(self, file_path: str) -> str:
        # Giữ tương thích cũ: trả string phẳng
        elements = self.load_structured(file_path)
        parts = []
        for el in elements:
            if el["type"] == "paragraph":
                parts.append(el["text"])
            elif el["type"] == "table":
                parts.append(el["text"])
        return "\n".join(parts)

    def load_structured(self, file_path: str):
        """Trả về list dict có chapter/heading/table_title, giữ thứ tự thực."""
        doc = docx.Document(file_path)
        elements = []

        current_chapter = ""
        current_heading = ""
        current_heading_level = 0
        pending_table_title = ""

        # Duyệt body theo thứ tự thực (paragraph xen table)
        for block in doc.element.body:
            if isinstance(block, CT_P):
                para = Paragraph(block, doc)
                text = para.text.strip()
                if not text:
                    continue
                style_name = para.style.name if para.style else ""

                # Heading detection theo Word style
                if style_name.startswith("Heading"):
                    try:
                        level = int(style_name.split()[-1])
                    except Exception:
                        level = 1
                    if level == 1:
                        # Chương: Heading 1
                        current_chapter = text
                        current_heading = text
                        current_heading_level = 1
                    else:
                        current_heading = text
                        current_heading_level = level
                    # Lưu heading như paragraph nhưng đánh dấu type heading
                    elements.append({
                        "type": "paragraph",
                        "text": text,
                        "chapter": current_chapter,
                        "heading": current_heading,
                        "heading_level": level,
                        "is_heading": True
                    })
                    # Heading không phải table_title
                    pending_table_title = ""
                    continue

                # Table title detection: paragraph ngay trước bảng chứa "Bảng"/"Table"
                if re.search(r"^\s*Bảng\s*\d+|^\s*Table\s*\d+", text, re.IGNORECASE):
                    pending_table_title = text
                    # vẫn lưu paragraph này như heading cho bảng
                    # lưu luôn để context có tiêu đề
                    elements.append({
                        "type": "paragraph",
                        "text": text,
                        "chapter": current_chapter,
                        "heading": current_heading,
                        "heading_level": current_heading_level,
                        "is_heading": False
                    })
                    continue

                elements.append({
                    "type": "paragraph",
                    "text": text,
                    "chapter": current_chapter,
                    "heading": current_heading,
                    "heading_level": current_heading_level,
                    "is_heading": False
                })
                # paragraph thường không phải table title nếu không match
                if pending_table_title and not text.startswith("Bảng"):
                    # giữ pending cho bảng kế tiếp, không xóa
                    pass

            elif isinstance(block, CT_Tbl):
                tbl = Table(block, doc)
                rows_text = []
                for row in tbl.rows:
                    row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_text:
                        rows_text.append(" | ".join(row_text))
                if not rows_text:
                    continue
                table_text = "\n".join(rows_text)

                # Gán table_title đang pending (nếu có)
                table_title = pending_table_title
                # reset sau khi dùng cho bảng này
                pending_table_title = ""

                # Tách header row (dòng đầu) để lặp lại khi chia bảng dài
                header_row = rows_text[0] if rows_text else ""
                # Nếu không có title, thử lấy heading hiện tại làm title
                elements.append({
                    "type": "table",
                    "text": table_text,
                    "table_title": table_title,
                    "header_row": header_row,
                    "chapter": current_chapter,
                    "heading": current_heading,
                    "heading_level": current_heading_level,
                    "is_heading": False
                })

        return elements
