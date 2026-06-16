import json
import re
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


JUDGE_SYSTEM = """\
You are a rigorous evaluator for SkyWay Airlines' customer-service chatbot.
You are given a passenger QUESTION, a numbered list of EVALUATION_CRITERIA
derived from the SkyWay Customer Service Reference Manual, and the chatbot's
RESPONSE. Decide whether the RESPONSE is factually acceptable.

For each criterion, decide whether the RESPONSE satisfies it:
- Satisfied: the RESPONSE conveys or is consistent with the required fact,
  condition, or behaviour described by the criterion.
- Not satisfied: the RESPONSE contradicts the criterion, omits the required
  information, or fabricates content that violates it.

The RESPONSE is acceptable only if ALL criteria are satisfied.

Not relevant to your verdict: tone, politeness, formatting, ordering, length,
or extra correct detail that does not contradict any criterion. Judge facts
only. Minor rounding or paraphrasing of the same fact is fine.

Reply with strict JSON only, no prose, no code fences:
{
  "criteria_verdicts": [
    {"index": 1, "satisfied": true|false, "reason": "<one short phrase>"},
    ...
  ],
  "acceptable_answer": true|false,
  "reason": "<if acceptable: one short sentence saying why; if not: list the violated criteria and provide a brief explanation for each criterion as to how it was violated.>"
}"""


def build_judge_llm(
    model: str = "openai/gpt-oss-120b:free", openrouter_api_key: str = ""
):
    return ChatOpenAI(
        model=model,
        base_url=OPENROUTER_BASE_URL,
        api_key=SecretStr(openrouter_api_key),
    )


def _extract_json(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in judge response: {text!r}")
    return json.loads(match.group(0))


def judge_response(
    question: str,
    evaluation_criteria: List[str],
    response: str,
    llm,
) -> Dict[str, object]:
    criteria_text = "\n".join(evaluation_criteria)
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"EVALUATION_CRITERIA:\n{criteria_text}\n\n"
        f"RESPONSE:\n{response}"
    )
    reply = llm.invoke(
        [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=user_msg)]
    )
    raw = reply.content if hasattr(reply, "content") else str(reply)
    try:
        parsed = _extract_json(raw)
        return {
            "acceptable_answer": bool(parsed["acceptable_answer"]),
            "reason": str(parsed.get("reason", "")).strip(),
            "criteria_verdicts": parsed.get("criteria_verdicts", []),
        }
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "acceptable_answer": None,
            "reason": f"judge_parse_error: {exc}; raw={raw!r}",
            "criteria_verdicts": [],
        }
