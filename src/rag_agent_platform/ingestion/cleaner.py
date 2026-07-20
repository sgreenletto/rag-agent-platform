"""Text cleaning boundary for document ingestion."""

import re


class TextCleaner:
    """Normalize extracted text before chunking."""

    def clean(self, text: str) -> str:
        """Return normalized text while preserving paragraph boundaries."""
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"([A-Za-z]+)-\n([A-Za-z]+)", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
