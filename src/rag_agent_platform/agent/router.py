"""Query analysis and semantics-preserving query rewriting services."""

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag_agent_platform.llm import ChatModel, extract_json_object
from rag_agent_platform.models import QueryType, RetrievedChunk


@dataclass(slots=True)
class QueryAnalysis:
    query_type: QueryType
    needs_retrieval: bool
    reason: str
    warning: str | None = None


@runtime_checkable
class QueryAnalyzer(Protocol):
    def analyze(self, query: str) -> QueryAnalysis:
        """Classify one query without selecting a manual retrieval mode."""
        ...


class StructuredQueryAnalyzer:
    """Prefer structured LLM classification and fall back to explicit rules."""

    _CHAT_PHRASES = ("你好", "您好", "你是谁", "你能做什么", "hello", "hi")
    _RELATION_WORDS = ("关系", "关联", "分别负责", "上下游", "影响路径", "联系", "依赖")
    _COMPLEX_WORDS = ("结合", "比较", "综合", "分别", "根据以上", "对比", "多个", "同时")

    def __init__(self, chat_model: ChatModel | None = None) -> None:
        self._chat_model = chat_model

    def analyze(self, query: str) -> QueryAnalysis:
        if self._chat_model is None:
            return self._rule_analysis(query)
        prompt = (
            "将问题分类为 simple、complex、relation、chat。relation 只表示用户明确询问实体关系、"
            "上下游或影响路径；complex 表示金额加流程、多条件、比较或综合；chat 表示普通对话。"
            "不得因为出现企业名称就自动判为 relation。只输出 JSON："
            '{"query_type": str, "needs_retrieval": bool, "reason": str}。\n'
            f"问题：{query}"
        )
        try:
            payload = extract_json_object(self._chat_model.invoke(prompt))
            query_type = QueryType(str(payload["query_type"]).lower())
            needs_retrieval = payload["needs_retrieval"]
            reason = payload["reason"]
            if not isinstance(needs_retrieval, bool) or not isinstance(reason, str):
                raise TypeError("classification response has invalid fields")
            if query_type is QueryType.CHAT:
                needs_retrieval = False
            rule_guard = self._rule_analysis(query)
            if (
                query_type is QueryType.RELATION and rule_guard.query_type is not QueryType.RELATION
            ) or (query_type is QueryType.SIMPLE and rule_guard.query_type is QueryType.COMPLEX):
                rule_guard.reason = (
                    f"规则守卫阻止 LLM 策略漂移（模型={query_type.value}）：{rule_guard.reason}"
                )
                return rule_guard
            return QueryAnalysis(query_type, needs_retrieval, reason)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            fallback = self._rule_analysis(query)
            fallback.warning = f"LLM 分类失败，已使用规则兜底：{type(exc).__name__}: {exc}"
            return fallback

    def _rule_analysis(self, query: str) -> QueryAnalysis:
        normalized = query.strip()
        lowered = normalized.lower()
        if any(phrase == lowered or phrase in lowered for phrase in self._CHAT_PHRASES):
            return QueryAnalysis(QueryType.CHAT, False, "规则识别为普通对话")
        if any(word in normalized for word in self._RELATION_WORDS):
            return QueryAnalysis(QueryType.RELATION, True, "规则命中关系类关键词")
        condition_count = sum(word in normalized for word in self._COMPLEX_WORDS)
        amount_workflow = _contains_amount(normalized) and any(
            word in normalized for word in ("流程", "审批", "审核", "采购", "买")
        )
        if (
            len(normalized) >= 60
            or condition_count > 0
            or normalized.count("，") >= 2
            or amount_workflow
        ):
            return QueryAnalysis(QueryType.COMPLEX, True, "规则识别为长问题或多条件流程问题")
        return QueryAnalysis(QueryType.SIMPLE, True, "规则默认识别为简单知识问题")


@runtime_checkable
class QueryRewriter(Protocol):
    def rewrite(
        self,
        *,
        original_query: str,
        current_query: str,
        evaluation_reason: str,
        chunks: list[RetrievedChunk],
        suggested_query: str | None,
    ) -> str:
        """Return a validated, semantics-preserving retrieval query."""
        ...


class BoundedQueryRewriter:
    """Conservatively normalize a query and reject semantic drift programmatically."""

    def __init__(self, chat_model: ChatModel | None = None) -> None:
        self._chat_model = chat_model
        self.last_validation_reason = ""

    def rewrite(
        self,
        *,
        original_query: str,
        current_query: str,
        evaluation_reason: str,
        chunks: list[RetrievedChunk],
        suggested_query: str | None,
    ) -> str:
        del chunks  # Evidence content must not leak new entities into the retrieval query.
        candidates: list[tuple[str, str]] = []
        suggestion = (suggested_query or "").strip()
        if suggestion and suggestion != current_query.strip():
            candidates.append(("evaluator suggestion", suggestion))
        if self._chat_model is not None:
            prompt = (
                "请对检索查询做语义守恒的规范化，只允许把口语改为正式表达、保留数字、实体、"
                "动作与目标并加入同义表达。不得添加新公司、品牌、产品型号、许可证、授权、"
                "供应商资质或用户未问的子问题；不得把流程问题改成关系问题。只输出一行查询。\n"
                f"原始问题：{original_query}\n当前查询：{current_query}\n"
                f"失败原因（仅作诊断，不得复制其中的新实体）：{evaluation_reason}"
            )
            try:
                rewritten = self._chat_model.invoke(prompt).strip()
                if rewritten and rewritten != current_query.strip():
                    candidates.append(("model rewrite", rewritten))
            except RuntimeError:
                pass

        fallback = _conservative_rule_rewrite(original_query)
        if fallback == current_query.strip():
            fallback = _add_safe_retrieval_terms(fallback)
        candidates.append(("rule fallback", fallback))
        failures: list[str] = []
        for source, candidate in candidates:
            valid, reason = validate_query_rewrite(original_query, candidate)
            if valid:
                self.last_validation_reason = f"accepted {source}: {reason}"
                return candidate.strip().strip('"“”')
            failures.append(f"rejected {source}: {reason}")
        self.last_validation_reason = "; ".join(failures) or "no usable rewrite"
        return original_query.strip()


_LATIN_NAME_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9_.+-]*\b")
_ORGANIZATION_PATTERN = re.compile(
    r"[\u4e00-\u9fff]{2,24}(?:股份有限公司|有限责任公司|有限公司|集团|公司)"
)
_SEMANTIC_EXPANSION_TERMS = (
    "商业",
    "开源",
    "SaaS",
    "云服务",
    "许可证",
    "授权",
    "代理",
    "资质",
    "销售",
    "供应商",
    "品牌",
    "型号",
)


def validate_query_rewrite(original_query: str, candidate: str) -> tuple[bool, str]:
    """Reject rewrites that do not preserve numbers, objects or named entities."""
    original = original_query.strip()
    rewritten = candidate.strip().strip('"“”')
    if not rewritten:
        return False, "rewrite is empty"
    if "\n" in rewritten:
        return False, "rewrite must be one line"
    original_amounts = _canonical_amounts(original)
    rewritten_amounts = _canonical_amounts(rewritten)
    if original_amounts and not original_amounts.issubset(rewritten_amounts):
        return False, "original numeric amount was not preserved"

    original_latin = {item.casefold() for item in _LATIN_NAME_PATTERN.findall(original)}
    new_latin = {
        item
        for item in _LATIN_NAME_PATTERN.findall(rewritten)
        if item.casefold() not in original_latin
    }
    if new_latin:
        return False, f"new Latin proper name: {sorted(new_latin)[0]}"
    original_organizations = set(_ORGANIZATION_PATTERN.findall(original))
    new_organizations = set(_ORGANIZATION_PATTERN.findall(rewritten)) - original_organizations
    if new_organizations:
        return False, f"new organization: {sorted(new_organizations)[0]}"
    for term in _SEMANTIC_EXPANSION_TERMS:
        if term.casefold() in rewritten.casefold() and term.casefold() not in original.casefold():
            return False, f"new business scope: {term}"

    if "数据库" in original and "数据库" not in rewritten:
        return False, "database object was dropped"
    if any(term in original for term in ("买", "采购", "购买")) and not any(
        term in rewritten for term in ("买", "采购", "购买")
    ):
        return False, "procurement action was dropped"
    if any(term in original for term in ("打钱", "付款")) and "付款" not in rewritten:
        return False, "payment goal was dropped"
    original_questions = original.count("？") + original.count("?")
    rewritten_questions = rewritten.count("？") + rewritten.count("?")
    if rewritten_questions > max(1, original_questions):
        return False, "rewrite added multiple questions"
    return True, "numbers, entities and user intent are conserved"


def _conservative_rule_rewrite(query: str) -> str:
    rewritten = query.strip()
    rewritten = re.sub(
        r"([零〇一二两三四五六七八九十百千万亿]+)块",
        lambda match: f"{_chinese_number_to_int(match.group(1))}元",
        rewritten,
    )
    rewritten = re.sub(r"(\d+(?:\.\d+)?)万块", r"\1万元", rewritten)
    rewritten = re.sub(r"(\d+(?:\.\d+)?)块", r"\1元", rewritten)
    replacements = (
        ("没预算", "未列入年度预算"),
        ("又很急的情况", "的紧急采购"),
        ("很急的情况", "紧急采购"),
        ("那种未列入", "未列入"),
        ("要找谁", "需要由谁批准"),
        ("东西", "采购物品"),
        ("送来以后", "交付后"),
        ("怎么才能打钱", "需要经过哪些验收和材料提交步骤才能付款"),
        ("怎么才能付款", "需要经过哪些验收和材料提交步骤才能付款"),
        ("打钱", "付款"),
        ("买", "采购"),
        ("那个流程怎么走", "时，需要经过哪些必要审核和审批流程"),
    )
    for source, target in replacements:
        rewritten = rewritten.replace(source, target)
    return rewritten.replace("？？", "？")


def _add_safe_retrieval_terms(query: str) -> str:
    stripped = query.rstrip("？?").strip()
    if any(term in stripped for term in ("流程", "审批", "审核", "批准", "付款")):
        return f"{stripped}，相关制度的适用条件是什么？"
    return f"{stripped}，相关制度如何规定？"


def _canonical_amounts(text: str) -> set[int]:
    amounts: set[int] = set()
    pattern = r"(\d+(?:\.\d+)?)\s*(万|千|百)?\s*(元|块)"
    for number, unit, _currency in re.findall(pattern, text):
        multiplier = {"": 1, "百": 100, "千": 1000, "万": 10000}[unit]
        amounts.add(int(float(number) * multiplier))
    for value in re.findall(r"[零〇一二两三四五六七八九十百千万亿]+(?=元|块)", text):
        amounts.add(_chinese_number_to_int(value))
    return amounts


def _chinese_number_to_int(value: str) -> int:
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}
    total = 0
    section = 0
    number = 0
    for char in value:
        if char in digits:
            number = digits[char]
            continue
        unit = units[char]
        if unit < 10000:
            section += (number or 1) * unit
        else:
            section = (section + number) * unit
            total += section
            section = 0
        number = 0
    return total + section + number


def _contains_amount(text: str) -> bool:
    return bool(_canonical_amounts(text))
