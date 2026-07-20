from rag_agent_platform.ingestion.cleaner import TextCleaner


def test_cleaner_strips_outer_whitespace() -> None:
    cleaned = TextCleaner().clean("  员工请假制度  \n")

    assert cleaned == "员工请假制度"


def test_cleaner_normalizes_newlines_and_spaces() -> None:
    dirty = "第一条\r\n\r\n\r\n第二条\t\t需要   审批"

    cleaned = TextCleaner().clean(dirty)

    assert cleaned == "第一条\n\n第二条需要审批"


def test_cleaner_removes_invisible_control_characters() -> None:
    dirty = "采购\x00流程\x08需要审批"

    cleaned = TextCleaner().clean(dirty)

    assert cleaned == "采购流程需要审批"


def test_cleaner_preserves_paragraph_boundaries() -> None:
    dirty = "第一段。\n\n第二段。"

    cleaned = TextCleaner().clean(dirty)

    assert cleaned == dirty


def test_cleaner_repairs_english_hyphen_line_breaks() -> None:
    dirty = "infor-\nmation management"

    cleaned = TextCleaner().clean(dirty)

    assert cleaned == "information management"


def test_cleaner_handles_empty_text() -> None:
    assert TextCleaner().clean("") == ""
