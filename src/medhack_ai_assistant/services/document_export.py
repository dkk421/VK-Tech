from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from html import escape
from io import BytesIO
import textwrap
from zipfile import ZIP_DEFLATED, ZipFile

from medhack_ai_assistant.domain.models import PatientExam


@dataclass(frozen=True)
class MedicalConclusionData:
    exam: PatientExam
    factors: str
    is_unfit: bool
    summary: str
    mkb_codes: tuple[str, ...] = ()


def generate_medical_docx(data: MedicalConclusionData) -> bytes:
    """Generate a lightweight DOCX medical conclusion without external dependencies."""
    paragraphs = _build_paragraphs(data)
    document_xml = _document_xml(paragraphs)

    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml())
        archive.writestr("_rels/.rels", _root_rels_xml())
        archive.writestr("docProps/core.xml", _core_xml())
        archive.writestr("docProps/app.xml", _app_xml())
        archive.writestr("word/_rels/document.xml.rels", _document_rels_xml())
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", _styles_xml())
    return buffer.getvalue()


def generate_medical_pdf(data: MedicalConclusionData) -> bytes:
    """Generate a PDF medical conclusion with PyMuPDF."""
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Для генерации PDF нужен pymupdf: uv add pymupdf") from exc

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    rect = fitz.Rect(42, 40, 553, 800)
    html = _medical_conclusion_html(data)

    try:
        page.insert_htmlbox(rect, html)
    except Exception:
        _draw_pdf_text_fallback(page, rect, data)

    output = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return output


def _medical_conclusion_html(data: MedicalConclusionData) -> str:
    exam = data.exam
    assigned = ";".join(exam.assigned_harmful_factors) if exam.assigned_harmful_factors else "0"
    mkb = "; ".join(data.mkb_codes) if data.mkb_codes else "не указаны"
    result_class = "bad" if data.is_unfit else "ok"
    result_text = (
        "ВЫЯВЛЕНЫ медицинские противопоказания к работе с вредными и (или) опасными факторами."
        if data.is_unfit
        else "НЕ ВЫЯВЛЕНЫ медицинские противопоказания к работе с вредными и (или) опасными факторами."
    )
    factor_label = "Противопоказанные пункты" if data.is_unfit else "Факторы по направлению"

    return f"""
<html>
<head>
<style>
body {{
  font-family: sans-serif;
  color: #111827;
  font-size: 11pt;
  line-height: 1.35;
}}
.title {{
  text-align: center;
  font-weight: 700;
  font-size: 13pt;
  margin-bottom: 18px;
}}
.section-title {{
  font-weight: 700;
  margin-top: 18px;
  margin-bottom: 8px;
}}
.row {{
  margin: 5px 0;
}}
.label {{
  font-weight: 700;
}}
.result {{
  padding: 10px 12px;
  border-radius: 6px;
  margin: 8px 0 12px 0;
  font-weight: 700;
}}
.ok {{
  color: #166534;
  background: #dcfce7;
}}
.bad {{
  color: #991b1b;
  background: #fee2e2;
}}
.signatures {{
  margin-top: 32px;
}}
</style>
</head>
<body>
  <div class="title">
    ЗАКЛЮЧЕНИЕ ПО РЕЗУЛЬТАТАМ ПЕРИОДИЧЕСКОГО МЕДИЦИНСКОГО ОСМОТРА<br/>
    по Приказу Минздрава РФ N 29н
  </div>

  <div class="row"><span class="label">ID пациента:</span> {escape(str(exam.patient_id))}</div>
  <div class="row"><span class="label">ID осмотра:</span> {escape(str(exam.exam_row_id))}</div>
  <div class="row"><span class="label">Дата консультации:</span> {escape(exam.consultation_date or "не указана")}</div>
  <div class="row"><span class="label">Направлен со следующими факторами:</span> {escape(assigned)}</div>

  <div class="section-title">Результат медицинского осмотра:</div>
  <div class="result {result_class}">{escape(result_text)}</div>

  <div class="row"><span class="label">{escape(factor_label)}:</span> {escape(data.factors)}</div>
  <div class="row"><span class="label">Коды МКБ из заключений:</span> {escape(mkb)}</div>
  <div class="row"><span class="label">Краткое обоснование:</span> {escape(data.summary)}</div>

  <div class="signatures">
    <div class="row">Председатель врачебной комиссии: __________________</div>
    <div class="row">Врач-профпатолог: __________________</div>
    <div class="row">Дата формирования: {date.today().isoformat()}</div>
  </div>
</body>
</html>
"""


def _draw_pdf_text_fallback(page, rect, data: MedicalConclusionData) -> None:
    lines = []
    for text, _ in _build_paragraphs(data):
        if not text:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(text, width=82) or [""])

    page.insert_textbox(
        rect,
        "\n".join(lines),
        fontsize=10,
        fontname="helv",
        align=0,
    )


def _build_paragraphs(data: MedicalConclusionData) -> list[tuple[str, bool]]:
    exam = data.exam
    result_text = (
        "ВЫЯВЛЕНЫ медицинские противопоказания к работе с вредными и (или) опасными факторами."
        if data.is_unfit
        else "НЕ ВЫЯВЛЕНЫ медицинские противопоказания к работе с вредными и (или) опасными факторами."
    )
    factor_label = "Противопоказанные пункты" if data.is_unfit else "Факторы по направлению"
    assigned = ";".join(exam.assigned_harmful_factors) if exam.assigned_harmful_factors else "0"
    mkb = "; ".join(data.mkb_codes) if data.mkb_codes else "не указаны"

    return [
        ("ЗАКЛЮЧЕНИЕ ПО РЕЗУЛЬТАТАМ ПЕРИОДИЧЕСКОГО МЕДИЦИНСКОГО ОСМОТРА", True),
        ("по Приказу Минздрава РФ N 29н", True),
        ("", False),
        (f"ID пациента: {exam.patient_id}", False),
        (f"ID осмотра: {exam.exam_row_id}", False),
        (f"Дата консультации: {exam.consultation_date or 'не указана'}", False),
        (f"Направлен со следующими факторами: {assigned}", False),
        ("", False),
        ("Результат медицинского осмотра:", True),
        (result_text, True),
        (f"{factor_label}: {data.factors}", False),
        (f"Коды МКБ из заключений: {mkb}", False),
        (f"Краткое обоснование: {data.summary}", False),
        ("", False),
        ("Председатель врачебной комиссии: __________________", False),
        ("Врач-профпатолог: __________________", False),
        (f"Дата формирования: {date.today().isoformat()}", False),
    ]


def _document_xml(paragraphs: list[tuple[str, bool]]) -> str:
    body = "\n".join(_paragraph_xml(text, bold=bold) for text, bold in paragraphs)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="708" w:footer="708" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>"""


def _paragraph_xml(text: str, *, bold: bool) -> str:
    if not text:
        return "<w:p/>"
    bold_xml = "<w:b/>" if bold else ""
    return f"""<w:p>
  <w:r>
    <w:rPr>
      <w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/>
      {bold_xml}
      <w:sz w:val="22"/>
    </w:rPr>
    <w:t>{escape(text)}</w:t>
  </w:r>
</w:p>"""


def _content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""


def _root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def _document_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr>
      <w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/>
      <w:sz w:val="22"/>
    </w:rPr>
  </w:style>
</w:styles>"""


def _core_xml() -> str:
    today = date.today().isoformat()
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Medical conclusion</dc:title>
  <dc:creator>MedHack AI Assistant</dc:creator>
  <cp:lastModifiedBy>MedHack AI Assistant</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{today}T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{today}T00:00:00Z</dcterms:modified>
</cp:coreProperties>"""


def _app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>MedHack AI Assistant</Application>
</Properties>"""
