"""Layout-aware extraction: a ``Source`` → ``StructuredDocument`` seam, the built-in extractors, and
the ``LoadAndParse`` stage that drives them.

Importing this package registers the built-in extractors *and* the ``LoadAndParse`` stage with the
global ``ComponentFactory`` (each self-registers under its ``class_name``), so ``Pipeline.from_spec`` /
``ComponentFactory`` can build them. ``LoadAndParseStage`` is imported last: it pulls in the ``extractor``
submodule, which the lines above have already loaded.
"""

from tarnrag.ingestion.components.extraction.docling_pdf import DoclingExtractor
from tarnrag.ingestion.components.extraction.extractor import Extractor, Source
from tarnrag.ingestion.components.extraction.html import HtmlExtractor
from tarnrag.ingestion.components.extraction.load_parse import LoadAndParseStage
from tarnrag.ingestion.components.extraction.markdown import MarkdownExtractor
from tarnrag.ingestion.components.extraction.pdf import PdfTextExtractor
from tarnrag.ingestion.components.extraction.plain_text import PlainTextExtractor
from tarnrag.ingestion.components.extraction.table_json import TableJsonExtractor

__all__ = [
    "DoclingExtractor",
    "Extractor",
    "HtmlExtractor",
    "LoadAndParseStage",
    "MarkdownExtractor",
    "PdfTextExtractor",
    "PlainTextExtractor",
    "Source",
    "TableJsonExtractor",
]
