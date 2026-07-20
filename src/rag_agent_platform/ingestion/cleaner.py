"""Text cleaning boundary for document ingestion."""


class TextCleaner:
    """Normalize extracted text before chunking.

    The first skeleton keeps behavior conservative. Detailed cleaning rules are
    implemented in the dedicated cleaner step.
    """

    def clean(self, text: str) -> str:
        """Return lightly normalized text."""
        return text.strip()
