"""User-centred grounded-answer evaluation and workflow decisions."""

import re

from rag_agent_platform.evaluation.base import (
    AnswerEvaluator,
    EvaluationDecision,
    EvaluationResult,
)
from rag_agent_platform.llm import ChatModel, extract_json_object
from rag_agent_platform.models import RetrievedChunk
from rag_agent_platform.responses import INSUFFICIENT_ANSWER
from rag_agent_platform.text import tokenize

_CITATION_PATTERN = re.compile(r"\[(\d+)]")
_LATIN_NAME_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9_.+-]*\b")
_ORGANIZATION_PATTERN = re.compile(
    r"[\u4e00-\u9fff]{2,24}(?:股份有限公司|有限责任公司|有限公司|集团|公司)"
)
_NEGATION_MARKERS = ("未说明", "没有说明", "无法确认", "不能确认", "不代表", "不等于")
_RISKY_CAPABILITY_TERMS = ("销售", "经销", "代理", "授权", "许可证", "资质")
_CLASSIFICATION_TERMS = ("小额采购", "普通采购", "重要采购", "重大采购")
_APPROVER_TERMS = (
    "部门负责人",
    "财务部门",
    "信息安全部门",
    "行政部门",
    "采购部门",
    "申请部门",
    "分管副总经理",
    "总经理",
    "直属负责人",
    "人力资源部门",
)
_STOP_WORDS = {
    "公司",
    "是否",
    "什么",
    "怎么",
    "如何",
    "哪些",
    "那个",
    "那种",
    "以后",
    "才能",
    "需要",
    "情况",
    "存在",
    "一个",
    "费用",
}


class GroundedAnswerEvaluator(AnswerEvaluator):
    """Separate answer defects from retrieval defects without expanding user intent."""

    def __init__(self, chat_model: ChatModel | None = None) -> None:
        self._chat_model = chat_model

    def evaluate(
        self,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> EvaluationResult:
        local_result = self._local_evaluation(query, answer, chunks)
        if local_result.decision is not EvaluationDecision.PASS or self._chat_model is None:
            return local_result

        context = "\n".join(f"[{i}] {chunk.content}" for i, chunk in enumerate(chunks, 1))
        prompt = self._build_prompt(query, answer, context)
        try:
            payload = extract_json_object(self._chat_model.invoke(prompt))
            model_result = self._parse_model_result(payload, local_result)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return EvaluationResult(
                False,
                f"模型评估失败；证据仍保留，先尝试约束性重新生成：{type(exc).__name__}: {exc}",
                decision=EvaluationDecision.REGENERATE,
                relevance_score=local_result.relevance_score,
                groundedness_score=local_result.groundedness_score,
                completeness_score=local_result.completeness_score,
                citation_quality_score=local_result.citation_quality_score,
            )
        return self._reconcile_model_decision(model_result, local_result)

    def _local_evaluation(
        self,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> EvaluationResult:
        if not chunks:
            return EvaluationResult(
                False,
                "知识库没有返回相关证据，无法形成有依据的回答",
                decision=EvaluationDecision.REFUSE,
            )

        relevance, requirements_met, missing_requirements = _evidence_coverage(query, chunks)
        if relevance < 0.3 and not requirements_met:
            return EvaluationResult(
                False,
                "检索证据与用户问题无关，知识库中没有可靠依据",
                decision=EvaluationDecision.REFUSE,
                relevance_score=relevance,
            )
        if missing_requirements:
            return EvaluationResult(
                False,
                "检索证据缺少回答核心问题所需的依据：" + "、".join(missing_requirements),
                query,
                decision=EvaluationDecision.REWRITE_RETRIEVE,
                relevance_score=relevance,
                groundedness_score=1.0,
                completeness_score=0.3,
                citation_quality_score=1.0,
            )

        markers = {int(value) for value in _CITATION_PATTERN.findall(answer)}
        valid_markers = set(range(1, len(chunks) + 1))
        if not markers:
            return EvaluationResult(
                False,
                "检索证据充分，但回答缺少知识库引用",
                decision=EvaluationDecision.REGENERATE,
                relevance_score=relevance,
                groundedness_score=0.5,
                completeness_score=0.8,
                citation_quality_score=0.0,
            )
        if not markers.issubset(valid_markers):
            return EvaluationResult(
                False,
                "检索证据充分，但回答包含不存在的引用编号",
                decision=EvaluationDecision.REGENERATE,
                relevance_score=relevance,
                groundedness_score=0.5,
                completeness_score=0.8,
                citation_quality_score=0.0,
            )

        answer_anomalies = _find_answer_anomalies(query, answer, chunks)
        if answer_anomalies:
            return EvaluationResult(
                False,
                "检索证据充分，但回答存在大面积无依据改写或与问题无关的正文复述；"
                "应使用原证据重新生成",
                decision=EvaluationDecision.REGENERATE,
                relevance_score=relevance,
                groundedness_score=0.2,
                completeness_score=0.7,
                citation_quality_score=0.5,
                unsupported_claims=answer_anomalies,
            )

        unsupported_claims = _find_unsupported_claims(query, answer, chunks)
        if unsupported_claims:
            return EvaluationResult(
                False,
                "检索证据充分，但回答包含无依据事实；应使用原证据重新生成",
                decision=EvaluationDecision.REGENERATE,
                relevance_score=relevance,
                groundedness_score=0.2,
                completeness_score=0.8,
                citation_quality_score=0.7,
                unsupported_claims=unsupported_claims,
            )
        if answer.strip() == INSUFFICIENT_ANSWER:
            return EvaluationResult(
                False,
                "检索证据充分，但生成器错误输出了无答案回复",
                decision=EvaluationDecision.REGENERATE,
                relevance_score=relevance,
                groundedness_score=1.0,
                completeness_score=0.0,
                citation_quality_score=1.0,
            )
        return EvaluationResult(
            True,
            "本地校验通过：核心证据充分，回答引用有效且未发现无依据扩展",
            decision=EvaluationDecision.PASS,
            relevance_score=relevance,
            groundedness_score=1.0,
            completeness_score=1.0,
            citation_quality_score=1.0,
        )

    @staticmethod
    def _build_prompt(query: str, answer: str, context: str) -> str:
        return (
            "你是严格、但以用户实际问题为中心的答案评估器。不得给用户增加未询问的验收项。\n"
            "只比较回答与上下文：轻微同义转述应接受；同一文档的多个不同证据块可以共同引用；"
            "不要因为来源相同或用户未区分开源、商业、SaaS、许可证而判失败。\n"
            "决策规则：\n"
            "- pass：核心答案有明确证据，引用支持关键结论。\n"
            "- regenerate：证据足够，但回答加入推断、供应商能力、品牌、资质，或引用错位。\n"
            "- rewrite_retrieve：证据确实缺少回答核心问题所需条款或明显偏题。\n"
            "- clarify：缺少关键前提会产生互相矛盾的答案；文档已有明确制度时不得使用。\n"
            "- refuse：知识库完全没有相关依据。\n"
            "维护服务不能推导为软件销售；信息技术供应商不能推导为任意产品供应商。"
            "只输出 JSON，字段为 decision、reason、relevance_score、groundedness_score、"
            "completeness_score、citation_quality_score、suggested_query、unsupported_claims。"
            "四个分数均为 0 到 1；unsupported_claims 是字符串数组。\n"
            f"用户问题：{query}\n回答：{answer}\n上下文：\n{context}"
        )

    @staticmethod
    def _parse_model_result(
        payload: dict[str, object],
        local_result: EvaluationResult,
    ) -> EvaluationResult:
        raw_decision = payload.get("decision")
        unsupported = payload.get("unsupported_claims", [])
        if not isinstance(unsupported, list) or any(
            not isinstance(item, str) for item in unsupported
        ):
            raise TypeError("unsupported_claims must be list[str]")
        if raw_decision is None:
            passed = payload.get("passed")
            if not isinstance(passed, bool):
                raise TypeError("evaluation response must include decision or passed")
            if passed:
                decision = EvaluationDecision.PASS
            elif unsupported:
                decision = EvaluationDecision.REGENERATE
            elif payload.get("suggested_query"):
                decision = EvaluationDecision.REWRITE_RETRIEVE
            else:
                decision = EvaluationDecision.REGENERATE
        else:
            decision = EvaluationDecision(str(raw_decision).lower())
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise TypeError("evaluation response has invalid reason")
        suggested_query = payload.get("suggested_query")
        if suggested_query is not None and not isinstance(suggested_query, str):
            raise TypeError("suggested_query must be a string or null")

        def score(name: str, fallback: float) -> float:
            value = payload.get(name, fallback)
            if not isinstance(value, int | float):
                raise TypeError(f"{name} must be numeric")
            return max(0.0, min(1.0, float(value)))

        return EvaluationResult(
            decision is EvaluationDecision.PASS,
            reason.strip(),
            suggested_query.strip() if isinstance(suggested_query, str) else None,
            decision=decision,
            relevance_score=score("relevance_score", local_result.relevance_score),
            groundedness_score=score("groundedness_score", local_result.groundedness_score),
            completeness_score=score("completeness_score", local_result.completeness_score),
            citation_quality_score=score(
                "citation_quality_score", local_result.citation_quality_score
            ),
            unsupported_claims=list(unsupported),
        )

    @staticmethod
    def _reconcile_model_decision(
        model_result: EvaluationResult,
        local_result: EvaluationResult,
    ) -> EvaluationResult:
        if model_result.decision is EvaluationDecision.PASS:
            return model_result
        if model_result.unsupported_claims:
            model_result.decision = EvaluationDecision.REGENERATE
            model_result.passed = False
            model_result.suggested_query = None
            return model_result
        if (
            model_result.decision
            in {
                EvaluationDecision.REWRITE_RETRIEVE,
                EvaluationDecision.REFUSE,
                EvaluationDecision.CLARIFY,
            }
            and local_result.relevance_score >= 0.8
            and local_result.completeness_score >= 0.8
        ):
            model_result.decision = EvaluationDecision.REGENERATE
            model_result.passed = False
            model_result.suggested_query = None
            model_result.reason = (
                "程序化证据覆盖检查显示检索充分；将模型的过严结论改为使用原证据重新生成。"
                + model_result.reason
            )
        return model_result


def _evidence_coverage(
    query: str,
    chunks: list[RetrievedChunk],
) -> tuple[float, bool, list[str]]:
    evidence = "\n".join(chunk.content for chunk in chunks)
    normalized_query = _normalize_query_terms(query)
    normalized_evidence = _normalize_query_terms(evidence)
    query_tokens = {
        token for token in tokenize(normalized_query) if token not in _STOP_WORDS and len(token) > 1
    }
    evidence_tokens = set(tokenize(normalized_evidence))
    lexical_score = (
        len(query_tokens.intersection(evidence_tokens)) / len(query_tokens) if query_tokens else 0.0
    )

    requirements: list[tuple[str, bool]] = []
    if _contains_amount(query) and any(term in normalized_query for term in ("采购", "购买", "买")):
        requirements.append(
            ("采购金额分级规则", "采购金额" in evidence and "超过" in evidence and "元" in evidence)
        )
    if "数据库" in query:
        requirements.append(("数据库软件相关条款", "数据库" in evidence))
    if any(term in query for term in ("流程", "怎么走", "如何办理")):
        requirements.append(
            ("办理或审批流程", any(term in evidence for term in ("流程", "审批", "审核")))
        )
    if any(term in query for term in ("很急", "紧急")):
        requirements.append(("紧急采购规则", "紧急采购" in evidence))
    if any(term in query for term in ("没预算", "未列入年度预算")):
        requirements.append(("未列入预算规则", "未列入年度预算" in evidence))
    if any(term in query for term in ("打钱", "付款")):
        requirements.append(("付款规则", "付款" in evidence))
    if any(term in query for term in ("送来", "交付后")):
        requirements.append(("交付验收规则", "交付" in evidence and "验收" in evidence))

    missing = [name for name, present in requirements if not present]
    requirement_score = (
        sum(present for _, present in requirements) / len(requirements) if requirements else 0.0
    )
    relevance = max(lexical_score, requirement_score)
    return relevance, bool(requirements) and not missing, missing


def _normalize_query_terms(text: str) -> str:
    replacements = {
        "买": "采购",
        "打钱": "付款",
        "送来": "交付",
        "没预算": "未列入年度预算",
        "很急": "紧急采购",
    }
    normalized = text
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def _contains_amount(text: str) -> bool:
    return bool(
        re.search(r"\d+(?:\.\d+)?\s*(?:元|块|万|千|百)", text)
        or re.search(r"[零〇一二两三四五六七八九十百千万亿]+(?:元|块|万)", text)
    )


def _find_unsupported_claims(
    query: str,
    answer: str,
    chunks: list[RetrievedChunk],
) -> list[str]:
    evidence = "\n".join(chunk.content for chunk in chunks)
    allowed_text = f"{query}\n{evidence}"
    unsupported: list[str] = []
    allowed_latin = {item.casefold() for item in _LATIN_NAME_PATTERN.findall(allowed_text)}
    for name in _LATIN_NAME_PATTERN.findall(answer):
        if name.casefold() not in allowed_latin:
            unsupported.append(f"上下文未出现的专有名词：{name}")

    allowed_organizations = set(_ORGANIZATION_PATTERN.findall(allowed_text))
    for organization in _ORGANIZATION_PATTERN.findall(answer):
        if organization not in allowed_organizations and not _has_supported_organization_suffix(
            organization, allowed_text
        ):
            unsupported.append(f"上下文未出现的企业：{organization}")

    sentences = [item.strip() for item in re.split(r"[。！？!?；;\n]", answer) if item.strip()]
    for sentence in sentences:
        cited_indexes = [int(value) for value in _CITATION_PATTERN.findall(sentence)]
        cited_text = "\n".join(
            chunks[index - 1].content for index in cited_indexes if 1 <= index <= len(chunks)
        )
        support_text = f"{query}\n{cited_text}"
        critical_terms = [term for term in _CLASSIFICATION_TERMS if term in sentence]
        critical_terms.extend(term for term in _APPROVER_TERMS if term in sentence)
        critical_terms.extend(re.findall(r"(?:至少)?[一二两三四五六七八九十\d]+家", sentence))
        critical_terms.extend(re.findall(r"[一二两三四五六七八九十\d]+个工作日", sentence))
        unsupported_terms = [term for term in critical_terms if term not in support_text]
        if cited_indexes and unsupported_terms:
            unsupported.append(f"引用未支持关键结论（{', '.join(unsupported_terms)}）：{sentence}")
        if any(marker in sentence for marker in _NEGATION_MARKERS):
            continue
        risky_terms = [term for term in _RISKY_CAPABILITY_TERMS if term in sentence]
        if not risky_terms:
            continue
        organizations = _ORGANIZATION_PATTERN.findall(sentence)
        if organizations:
            for organization in organizations:
                if not any(
                    organization in chunk.content
                    and any(term in chunk.content for term in risky_terms)
                    for chunk in chunks
                ):
                    unsupported.append(f"无依据的企业能力断言：{sentence}")
        elif any(term not in allowed_text for term in risky_terms):
            unsupported.append(f"上下文未支持的能力或资质：{sentence}")
    return list(dict.fromkeys(unsupported))


def _has_supported_organization_suffix(organization: str, allowed_text: str) -> bool:
    """Tolerate a role phrase greedily captured before an evidenced company name."""
    minimum_length = max(6, len(organization) // 2)
    return any(
        organization[start:] in allowed_text
        for start in range(0, len(organization) - minimum_length + 1)
    )


def _find_answer_anomalies(
    query: str,
    answer: str,
    chunks: list[RetrievedChunk],
) -> list[str]:
    """Detect unsupported semantic mutation and whole-parent evidence echoing."""
    evidence = "\n".join(chunk.content for chunk in chunks)
    allowed_text = f"{query}\n{evidence}"
    anomalies: list[str] = []
    summary_requested = any(
        term in query for term in ("全文", "完整内容", "逐条", "全部条款", "通篇", "原文")
    )

    if not summary_requested and len(answer) > 800:
        anomalies.append(f"回答长度异常（{len(answer)} 字），接近整篇文档复述")
    longest_chunk = max((len(chunk.content) for chunk in chunks), default=0)
    if (
        not summary_requested
        and longest_chunk >= 600
        and len(answer) >= 500
        and len(answer) >= longest_chunk * 0.65
    ):
        anomalies.append("回答长度接近检索父块，存在直接倾倒整段证据的风险")

    generic_answer_terms = {
        "回答",
        "文档",
        "制度",
        "材料",
        "规定",
        "明确",
        "当前",
        "知识库",
        "依据",
        "相关",
        "因此",
        "之后",
        "其中",
        "需要",
        "必须",
        "可以",
        "应当",
        "无法",
        "确认",
        "能力",
        "情况",
        "进行",
        "完成",
    }
    content_tokens = {
        token
        for token in tokenize(answer)
        if len(token) > 1
        and token not in _STOP_WORDS
        and not token.isdecimal()
        and not _CITATION_PATTERN.fullmatch(token)
    }
    unsupported_tokens = sorted(
        token
        for token in content_tokens
        if token not in generic_answer_terms and token not in allowed_text
    )
    if len(unsupported_tokens) >= 3 and (
        len(unsupported_tokens) / max(1, len(content_tokens)) >= 0.08
    ):
        anomalies.append(
            "回答包含多个问题和证据均未出现的内容词：" + "、".join(unsupported_tokens[:8])
        )

    if not summary_requested and len(answer) >= 350:
        focus_terms = _query_focus_terms(query)
        if focus_terms:
            sentences = [
                sentence.strip()
                for sentence in re.split(r"[。！？!?；;\n]", answer)
                if sentence.strip()
            ]
            relevant = sum(any(term in sentence for term in focus_terms) for sentence in sentences)
            if sentences and relevant / len(sentences) < 0.4:
                anomalies.append("回答复述了大量与用户实际问题无关的章节")
    return list(dict.fromkeys(anomalies))


def _query_focus_terms(query: str) -> tuple[str, ...]:
    if any(term in query for term in ("打钱", "付款", "交付", "送来", "验收")):
        return ("验收", "交付", "合同", "记录", "发票", "付款")
    if "数据库" in query:
        return ("数据库", "采购", "金额", "审核", "审批", "报价")
    if any(term in query for term in ("没预算", "未列入年度预算", "紧急")):
        return ("预算", "紧急", "批准", "审批", "补齐")
    if any(term in query for term in ("关系", "关联")):
        return ("负责", "联系", "签订", "提供", "关系", "关联")
    return tuple(token for token in tokenize(query) if len(token) > 1 and token not in _STOP_WORDS)
