import json
import re
from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


JUDGE_SYSTEM = (
    "You are evaluating whether a RAG system's response correctly answers a "
    "question, given the expected answer drawn from a company handbook."
    "Decide whether the RESPONSE is an acceptable answer to the QUESTION — "
    "it must convey the key facts present in the EXPECTED_ANSWER without "
    "contradicting them. Minor phrasing differences are fine. Missing key "
    "facts or inventing facts not supported by the expected answer is not "
    "fine."
    "Reply with strict JSON only, no prose, no code fences:"
    '{"acceptable_answer": true|false, "reason": "<one short sentence>"}'
)


def build_judge_llm(model: str = "openai/gpt-oss-120b:free", openrouter_api_key: str = ""):
    return ChatOpenAI(
        model=model,
        openai_api_base=OPENROUTER_BASE_URL,
        openai_api_key=openrouter_api_key,
    )


def _extract_json(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in judge response: {text!r}")
    return json.loads(match.group(0))


def judge_response(
    question: str,
    expected_answer: str,
    response: str,
    llm,
) -> Dict[str, object]:
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"EXPECTED_ANSWER:\n{expected_answer}\n\n"
        f"RESPONSE:\n{response}"
    )
    reply = llm.invoke([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=user_msg)])
    raw = reply.content if hasattr(reply, "content") else str(reply)
    try:
        parsed = _extract_json(raw)
        return {
            "acceptable_answer": bool(parsed["acceptable_answer"]),
            "reason": str(parsed.get("reason", "")).strip(),
        }
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "acceptable_answer": None,
            "reason": f"judge_parse_error: {exc}; raw={raw!r}",
        }
