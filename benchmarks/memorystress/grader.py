"""GPT-4o grading with question-type-aware prompts.

Reuses the LongMemEval grading pattern: GPT-4o judges whether the model's
response contains the correct answer. Different question types get specialized
grading prompts.
"""

from __future__ import annotations

from benchmarks.memorystress.llm import call_llm

# Question-type-aware grading prompts
GRADE_PROMPTS = {
    "default": """\
I will give you a question, a correct answer, and a response from a model. \
Please answer yes if the response contains the correct answer. Otherwise, answer no. \
If the response is equivalent to the correct answer or contains all the intermediate \
steps to get the correct answer, you should also answer yes. If the response only \
contains a subset of the information required by the answer, answer no.

Question: {question}
Correct Answer: {answer}
Model Response: {hypothesis}

Is the model response correct? Answer yes or no only.""",
    "contradiction_resolution": """\
I will give you a question, a correct answer, and a response from a model. \
The correct answer reflects ALL updates to a fact, including any reverts. \
The most recent value is the ground truth.

Please answer yes if the response contains the correct (most recent) answer. \
If the response gives an outdated value or mixes old and new values incorrectly, \
answer no. If the response identifies the update history AND gives the correct \
current value, answer yes.

Question: {question}
Correct Answer: {answer}
Model Response: {hypothesis}

Is the model response correct? Answer yes or no only.""",
    "temporal_ordering": """\
I will give you a question, a correct answer, and a response from a model. \
Please answer yes if the response contains the correct answer. Otherwise, answer no. \
Do not penalize off-by-one errors for the number of days. If the question \
asks for the number of days/weeks/months, and the model makes off-by-one errors \
(e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Question: {question}
Correct Answer: {answer}
Model Response: {hypothesis}

Is the model response correct? Answer yes or no only.""",
    "cross_agent_recall": """\
I will give you a question, a correct answer, and a response from a model. \
Please answer yes if the response contains the correct answer. Otherwise, answer no. \
The question asks about information from a different agent's sessions. The model \
should be able to recall cross-agent information if the memory system supports it.

Question: {question}
Correct Answer: {answer}
Model Response: {hypothesis}

Is the model response correct? Answer yes or no only.""",
    "cold_start_recall": """\
I will give you a question, a correct answer, and a response from a model. \
Please answer yes if the response contains the correct answer. Otherwise, answer no. \
The model was started fresh (cold start) but has access to an existing memory store. \
It should still be able to answer from persisted memories.

Question: {question}
Correct Answer: {answer}
Model Response: {hypothesis}

Is the model response correct? Answer yes or no only.""",
}


def grade_answer(
    question: dict,
    hypothesis: str,
    model: str = "gpt-4o",
    api_key: str | None = None,
) -> tuple[bool, int]:
    """Grade a hypothesis against a question's ground truth.

    Returns:
        (is_correct, tokens_used) — whether the answer is correct and
        approximate tokens consumed by the grading call.
    """
    question_type = question.get("question_type", "fact_recall")
    answer = question.get("answer", "")
    answer_detail = question.get("answer_detail", "")

    # Use answer_detail for grading context if available
    ground_truth = f"{answer}"
    if answer_detail:
        ground_truth += f" (Context: {answer_detail})"

    # Select prompt template
    template = GRADE_PROMPTS.get(question_type, GRADE_PROMPTS["default"])

    prompt = template.format(
        question=question.get("question", ""),
        answer=ground_truth,
        hypothesis=hypothesis,
    )

    result = call_llm(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=10,
        api_key=api_key,
    )
    is_correct = "yes" in result.lower()
    tokens_used = len(prompt) // 4 + 10  # rough estimate
    return is_correct, tokens_used
