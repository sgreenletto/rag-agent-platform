from rag_agent_platform.generation import GroundedAnswerGenerator
from rag_agent_platform.models import RetrievedChunk


class RecordingChatModel:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "数据库软件采购必须先经过信息安全部门审核 [1]。"


def evidence() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="security",
        content="数据库软件采购必须先经过信息安全部门审核。",
        normalized_score=0.9,
        source="policy.txt",
        retrieval_method="advanced",
    )


def test_generator_prompt_forbids_supplier_capability_inference() -> None:
    model = RecordingChatModel()
    generator = GroundedAnswerGenerator(model)

    generator.generate("八万块买数据库那个流程怎么走？", [evidence()])

    prompt = model.prompts[0]
    assert "信息技术供应商不等于" in prompt
    assert "数据库维护服务也不等于数据库软件销售" in prompt
    assert "文档未说明" in prompt
    assert "用户询问制度流程时，只回答制度明确要求的步骤" in prompt


def test_regeneration_uses_same_evidence_and_includes_unsupported_claim_feedback() -> None:
    model = RecordingChatModel()
    generator = GroundedAnswerGenerator(model)

    answer, citations = generator.regenerate(
        "八万块买数据库那个流程怎么走？",
        [evidence()],
        previous_answer="云帆可以销售数据库软件 [1]。",
        unsupported_claims=["无依据的企业能力断言"],
    )

    assert "信息安全部门审核" in answer
    assert citations[0].chunk_id == "security"
    assert "这是重新生成" in model.prompts[0]
    assert "无依据的企业能力断言" in model.prompts[0]
    assert "云帆可以销售数据库软件" in model.prompts[0]


def test_generator_context_is_bounded_and_keeps_query_relevant_local_sentence() -> None:
    model = RecordingChatModel()
    generator = GroundedAnswerGenerator(model, max_context_chars=1000)
    long_parent = RetrievedChunk(
        chunk_id="long-parent",
        content=("员工休假章节与当前问题无关。" * 500)
        + "采购物品交付后，财务部门收到完整材料，应当在十个工作日内完成付款。",
        normalized_score=0.9,
        source="policy.txt",
        retrieval_method="advanced",
    )

    generator.generate("东西送来以后怎么才能打钱？", [long_parent])

    prompt = model.prompts[0]
    assert "十个工作日内完成付款" in prompt
    assert prompt.count("员工休假章节") < 100
    assert len(prompt) < 10000
