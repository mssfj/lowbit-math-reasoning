# math_verify.py
"""
数学系タスク用の自動正解判定モジュール（math-verify の簡易実装）

用途：
- 評価パイプラインでの正答判定（EM / 数値 / SymPy）
- GRPO の reward 関数
- SFT / RL データのクリーニング

前提：
- gold_answer は「最終的な正解」を文字列で持つ
- pred_text は LLM の生出力（CoT込み）をそのまま渡してよい
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from sympy import simplify, sympify
from sympy.core.sympify import SympifyError


# =========================
# 正規化・抽出まわり
# =========================

_FINAL_ANSWER_PATTERNS = [
    r"final\s*answer\s*(?:[:：=]|is\b|->|⇒|=>)\s*(.+)",
    r"final\s*ans\s*(?:[:：=]|is\b|->|⇒|=>)\s*(.+)",
    r"(?:the\s+)?answer\s*(?:is\b|[:：=])\s*(.+)",
    r"最終解\s*(?:[:：=は]|です)\s*(.+)",
    r"最終答え\s*(?:[:：=は]|です)\s*(.+)",
    r"最終的な答えは\s*(?:[:：=])?\s*(.+)",
    r"答えは\s*(?:[:：=])?\s*(.+)",
    r"答え\s*(?:[:：=は]|です)\s*(.+)",
]


@dataclass
class ExtractedAnswer:
    answer: str
    has_final_answer: bool
    source: str


def _normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _strip_trailing_punct(s: str) -> str:
    return re.sub(r"[，,。．、!！?？]+$", "", s).strip()


def _strip_markdown_wrappers(s: str) -> str:
    s = s.strip()
    if (s.startswith("**") and s.endswith("**")) or (s.startswith("__") and s.endswith("__")):
        s = s[2:-2].strip()
    elif (s.startswith("*") and s.endswith("*")) or (s.startswith("_") and s.endswith("_")):
        s = s[1:-1].strip()
    elif s.startswith("`") and s.endswith("`"):
        s = s[1:-1].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    elif s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()

    s = re.sub(r"[`*_]+$", "", s).strip()
    s = re.sub(r"^[`*_]+", "", s).strip()
    return s


def _postprocess_candidate(s: str) -> str:
    s = _normalize_text(s)
    s = _strip_markdown_wrappers(s)
    return _strip_trailing_punct(s)


def _normalize_latex_expression(s: str) -> str:
    s = _postprocess_candidate(s)
    if not s:
        return s
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", " ")
    s = re.sub(r"\\text\s*\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^{}]+)\}", r"\1", s)
    s = s.replace("\\pi", "pi")
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\b([A-Za-z0-9.+-]+)\s*/\s*([A-Za-z0-9.+-]+)\b", r"(\1)/(\2)", s)
    s = s.replace("^\\circ", " degrees")
    s = s.replace("^{\\circ}", " degrees")
    s = s.replace("°", " degrees")
    s = re.sub(r"\bdegrees?\b", " degrees", s)
    s = re.sub(r'["\']+$', "", s)
    s = re.sub(r"\s*([(),=\[\]])\s*", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_reasoning_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text


def _is_instruction_like(s: str) -> bool:
    lowered = _normalize_text(s).lower()
    markers = (
        "the last line must be",
        "formatted as",
        "output the answer in the format",
        "do not output anything after",
        "ensure the final line",
        "final answer: ...",
        "final answer: <answer>",
        "final answer: <number>",
    )
    return any(marker in lowered for marker in markers)


_NUMERIC_TOKEN_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?")


def _extract_numeric_token(text: str) -> Optional[str]:
    m = _NUMERIC_TOKEN_RE.search(text)
    if m:
        return _postprocess_candidate(m.group(0))
    return None


def _looks_like_expression(text: str) -> bool:
    markers = ("\\frac", "\\sqrt", "\\pi", "(", ")", "[", "]", "{", "}", ",", "=")
    return any(marker in text for marker in markers)


def _looks_like_named_answer(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", text))


def _is_complete_expression(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(("=", "+", "-", "*", "/", "^", "_", "\\", "(", "[", "{", ",")):
        return False
    if stripped.count("(") != stripped.count(")"):
        return False
    if stripped.count("[") != stripped.count("]"):
        return False
    if stripped.count("{") != stripped.count("}"):
        return False
    return True


def _looks_like_standalone_answer(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if not _is_complete_expression(stripped):
        return False
    if len(stripped) > 64:
        return False
    if ":" in stripped:
        return False
    if _looks_like_named_answer(stripped):
        return True
    if _looks_like_expression(stripped) and len(stripped.split()) <= 4:
        return True
    if _extract_numeric_token(stripped) and _normalize_text(stripped) == _extract_numeric_token(stripped):
        return True
    return False


def _extract_boxed_from_line(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None

    start_idx = idx + len("\\boxed{")
    depth = 1
    for i in range(start_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                trailing = text[i + 1 :].strip()
                if trailing and trailing not in {".", "。"}:
                    return None
                return text[start_idx:i]
    return None


def extract_final_answer(raw_text: str) -> str:
    return extract_final_answer_with_meta(raw_text).answer


def extract_final_answer_with_meta(raw_text: str) -> ExtractedAnswer:
    text = _strip_reasoning_tags(raw_text).strip()
    if not text:
        return ExtractedAnswer("", False, "empty")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ExtractedAnswer("", False, "empty_lines")

    last_line = lines[-1]
    boxed_content = _extract_boxed_from_line(last_line)
    if boxed_content:
        candidate = _normalize_latex_expression(boxed_content)
        if candidate:
            return ExtractedAnswer(candidate, True, "boxed")

    if not _is_instruction_like(last_line):
        for pat in _FINAL_ANSWER_PATTERNS:
            m = re.search(pat, last_line, flags=re.IGNORECASE)
            if not m:
                continue
            candidate = _normalize_latex_expression(m.group(1))
            if not candidate or _is_instruction_like(candidate):
                continue
            if _looks_like_expression(candidate) or _looks_like_named_answer(candidate):
                return ExtractedAnswer(candidate, True, "pattern_line_expr")
            token = _extract_numeric_token(candidate)
            return ExtractedAnswer(token if token else candidate, True, "pattern_line")

    keyword_re = re.compile(r"(final answer|final ans|answer|最終解|最終答え|答え)", flags=re.IGNORECASE)
    if keyword_re.search(last_line) and not _is_instruction_like(last_line):
        candidate = _normalize_latex_expression(last_line)
        if candidate and not _is_instruction_like(candidate):
            if not _is_complete_expression(candidate):
                return ExtractedAnswer(candidate, False, "incomplete_keyword_line")
            if _looks_like_expression(candidate) or _looks_like_named_answer(candidate):
                return ExtractedAnswer(candidate, True, "keyword_line_expr")
            token = _extract_numeric_token(candidate)
            return ExtractedAnswer(token if token else candidate, True, "keyword_line")

    last = _normalize_latex_expression(lines[-1])
    if _looks_like_standalone_answer(last):
        if _looks_like_named_answer(last):
            has_final = len(lines) == 1 or len(last.split()) <= 3
            return ExtractedAnswer(last, has_final, "fallback_named")
        token = _extract_numeric_token(last)
        if token and _normalize_text(last) == token:
            bare_answer = len(lines) == 1 or _normalize_text(last) == token
            return ExtractedAnswer(token, bare_answer, "fallback")
        return ExtractedAnswer(last, len(lines) == 1, "fallback_expr")

    return ExtractedAnswer(last, False, "fallback")


# =========================
# 数値パース・比較
# =========================

def _parse_number(s: str) -> Optional[float]:
    s = _normalize_text(s)
    if not s:
        return None

    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return None

    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                num = float(parts[0])
                den = float(parts[1])
                if den == 0:
                    return None
                return num / den
            except ValueError:
                pass

    try:
        return float(s)
    except ValueError:
        return None


def numeric_close(a: float, b: float, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


# =========================
# SymPy を使った等価性チェック
# =========================

def sympy_equiv(pred: str, gold: str) -> bool:
    pred = pred.strip()
    gold = gold.strip()
    if not pred or not gold:
        return False

    try:
        ep = sympify(pred)
        eg = sympify(gold)
    except SympifyError:
        return False
    except Exception:
        return False

    try:
        diff = simplify(ep - eg)
        return diff == 0
    except Exception:
        return False


# =========================
# メイン verify ロジック
# =========================

@dataclass
class MathVerifyConfig:
    use_exact: bool = True
    use_numeric: bool = True
    use_sympy: bool = True
    rel_tol: float = 1e-6
    abs_tol: float = 1e-9
    require_final_answer: bool = True


@dataclass
class MathVerifyResult:
    is_correct: bool
    reason: str
    pred_answer: str
    gold_answer: str


def verify_math_answer(
    pred_text: str,
    gold_answer: str,
    config: Optional[MathVerifyConfig] = None,
) -> MathVerifyResult:
    if config is None:
        config = MathVerifyConfig()

    gold = _normalize_latex_expression(gold_answer)
    extracted = extract_final_answer_with_meta(pred_text)
    pred_raw = extracted.answer
    pred = _normalize_latex_expression(pred_raw)

    if config.require_final_answer and not extracted.has_final_answer:
        return MathVerifyResult(
            is_correct=False,
            reason="missing_final_answer",
            pred_answer=pred,
            gold_answer=gold,
        )

    if config.use_exact and pred == gold:
        return MathVerifyResult(
            is_correct=True,
            reason="exact_match",
            pred_answer=pred,
            gold_answer=gold,
        )

    if config.use_exact:
        pred_no_degrees = re.sub(r"\s+degrees\b", "", pred).strip()
        gold_no_degrees = re.sub(r"\s+degrees\b", "", gold).strip()
        if pred_no_degrees and pred_no_degrees == gold_no_degrees:
            return MathVerifyResult(
                is_correct=True,
                reason="exact_match",
                pred_answer=pred,
                gold_answer=gold,
            )

    if config.use_numeric:
        gv = _parse_number(gold)
        pv = _parse_number(pred)
        if gv is not None and pv is not None:
            if numeric_close(pv, gv, rel_tol=config.rel_tol, abs_tol=config.abs_tol):
                return MathVerifyResult(
                    is_correct=True,
                    reason="numeric_close",
                    pred_answer=pred,
                    gold_answer=gold,
                )

    if config.use_sympy:
        if sympy_equiv(pred, gold):
            return MathVerifyResult(
                is_correct=True,
                reason="sympy_equiv",
                pred_answer=pred,
                gold_answer=gold,
            )

    return MathVerifyResult(
        is_correct=False,
        reason="mismatch",
        pred_answer=pred,
        gold_answer=gold,
    )


# =========================
# RL 用の reward ラッパ
# =========================

def math_reward(
    pred_text: str,
    gold_answer: str,
    correct_reward: float = 1.0,
    wrong_reward: float = 0.0,
    config: Optional[MathVerifyConfig] = None,
) -> Tuple[float, MathVerifyResult]:
    result = verify_math_answer(pred_text, gold_answer, config=config)
    reward = correct_reward if result.is_correct else wrong_reward
    return reward, result
