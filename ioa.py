"""MCQ probing protocol for IoA-Suite."""

import json
import re
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent / "prompts"


def prompt(filename, **slots):
    text = (PROMPTS / filename).read_text(encoding="utf-8")
    if not slots:
        return text
    pattern = r"\{" + "(" + "|".join(re.escape(key) for key in slots) + ")" + r"\}"
    return re.sub(pattern, lambda match: slots[match[1]], text)


def public_profiles(participants):
    return "\n".join(
        f"- {p['name']}: {p['role']}\n"
        f"  Expertise: {p['expertise']}\n"
        f"  Background: {p['background']}"
        for p in participants
    )


def transcript(case, participant_id=None):
    lines = []
    for turn in case["dialogue"]["turns"]:
        line = f"[{turn['speaker_name']}]: "
        if participant_id is not None and turn["speaker_id"] == participant_id:
            line += f"(My thought: {turn['inner_thought']})\n  "
        lines.append(line + turn["utterance"])
    return "\n".join(lines)


def prober_messages(case):
    return prompt("prober.txt"), prompt(
        "prober_user.txt",
        member_profiles=public_profiles(case["profiles"]["members"]),
        transcript=transcript(case),
    )


def question_text(question_id, question):
    options = "\n".join(f"  ({key}) {value}" for key, value in question["options"].items())
    return f"Question {question_id}:\n{question['stem']}\n{options}"


def simulator_messages(case, participant, questions):
    others = [p for p in case["profiles"]["members"] if p["id"] != participant["id"]]
    view = (
        "<member_profiles>\n" + public_profiles(others) + "\n</member_profiles>\n\n"
        + transcript(case, participant["id"])
    )
    system = prompt(
        "simulator.txt",
        name=participant["name"], role=participant["role"],
        expertise=participant["expertise"], background=participant["background"],
        private_state=participant["private_state"],
        assumptions_block="\n".join(f"- {a}" for a in participant["hidden_assumptions"]),
        dialogue_view=view,
    )
    user = "Answer the following questions based on your understanding of the meeting.\n\n"
    user += "\n\n".join(question_text(i, q) for i, q in enumerate(questions))
    return system, user


def divergent_ids(questions, answers):
    return [
        i for i in range(len(questions))
        if len({participant[i]["choice"] for participant in answers}) > 1
    ]


def validity_filter_messages(case, questions, answers):
    participants = case["profiles"]["members"]
    ground_truth = "\n\n".join(
        f"{p['id']}: {p['topic']}\n"
        f"  Surface form: {p['surface_form']}\n"
        f"  {participants[0]['name']} understanding: {p['member_a_understanding']}\n"
        f"  {participants[1]['name']} understanding: {p['member_b_understanding']}\n"
        f"  Why undetected: {p['why_undetected']}"
        for p in case["profiles"]["misalignment_points"]
    )
    profiles = "\n\n".join(
        public_profiles([p]) + f"\n  Private State: {p['private_state']}\n"
        + "  Hidden Assumptions:\n" + "\n".join(f"    - {a}" for a in p["hidden_assumptions"])
        for p in participants
    )
    blocks = []
    for i in divergent_ids(questions, answers):
        block = question_text(i, questions[i])
        block += f"\n  Signal: {questions[i]['misalignment_signal']}"
        for participant, response in zip(participants, answers):
            block += f"\n  {participant['name']} answered: {response[i]['choice']}"
        blocks.append(block)
    user = "\n\n".join(
        f"<{tag}>\n{value}\n</{tag}>" for tag, value in (
            ("ground_truth", ground_truth), ("member_profiles", profiles),
            ("meeting_transcript", transcript(case)),
            ("diverging_questions", "\n\n".join(blocks)),
        )
    )
    return prompt("validity_filter.txt"), user


def parse_response(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The model returned no answer content.")
    text = text.strip()
    if not text.startswith(("{", "```")) and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")
    return result


def validate_questions(result):
    questions = result.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Prober response must contain a questions list.")
    for question in questions:
        if not isinstance(question, dict) or any(
            not isinstance(question.get(key), str) or not question[key].strip()
            for key in ("stem", "misalignment_signal")
        ):
            raise ValueError("Every question needs a non-empty stem and misalignment_signal.")
        options = question.get("options")
        if not isinstance(options, dict) or not 2 <= len(options) <= 4:
            raise ValueError("Every question needs 2–4 options.")
        if not set(options) <= set("ABCD") or any(
            not isinstance(value, str) or not value.strip() for value in options.values()
        ):
            raise ValueError("Options must map letters A–D to non-empty strings.")
    return questions


def indexed(items, expected_ids, label):
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError(f"{label} must be a list of objects.")
    ids = [item.get("question_id") for item in items]
    if any(type(i) is not int for i in ids) or len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError(f"{label} must cover each requested question_id exactly once.")
    return {item["question_id"]: item for item in items}


def validate_answers(result, questions):
    answers = indexed(result.get("answers"), range(len(questions)), "Answers")
    for i, answer in answers.items():
        if answer.get("choice") not in questions[i]["options"]:
            raise ValueError(f"Invalid option for question {i}.")
    return [answers[i] for i in range(len(questions))]


def validate_evaluations(result, ids, ground_truth_ids):
    evaluations = indexed(result.get("evaluations"), ids, "Validity filter evaluations")
    for i, evaluation in evaluations.items():
        for key in ("is_meaningful", "is_implicit"):
            if type(evaluation.get(key)) is not bool:
                raise ValueError(f"{key} must be a boolean for question {i}.")
        if "redundant_with" not in evaluation or "hits_ground_truth" not in evaluation:
            raise ValueError(f"Missing attribution or redundancy decision for question {i}.")
        earlier = evaluation["redundant_with"]
        if isinstance(earlier, str) and earlier.isdecimal():
            earlier = evaluation["redundant_with"] = int(earlier)
        if earlier is not None and (type(earlier) is not int or earlier not in ids or earlier >= i):
            raise ValueError(f"Question {i} must reference an earlier divergent question.")
        hit = evaluation["hits_ground_truth"]
        if hit is not None and (not isinstance(hit, str) or hit not in ground_truth_ids):
            raise ValueError(f"Unknown ground truth attribution for question {i}.")
    return evaluations


def score(questions, answers, filter_result, ground_truth):
    """Algorithm 1: exclude Category B; count unique ground truth points in recall."""
    ids = divergent_ids(questions, answers)
    gt_ids = {point["id"] for point in ground_truth}
    evaluations = {e["question_id"]: e for e in filter_result["evaluations"]}
    categories, hits, counts = [], set(), {"A": 0, "B": 0, "C": 0}
    for i in range(len(questions)):
        evaluation = evaluations.get(i)
        valid = i in ids and evaluation is not None and evaluation["is_meaningful"] is True and evaluation["is_implicit"] is True and evaluation.get("redundant_with") is None
        hit = evaluation.get("hits_ground_truth") if valid else None
        if hit not in gt_ids:
            hit = None
        category = ("A" if hit is not None else "B") if valid else "C"
        counts[category] += 1
        if hit is not None:
            hits.add(hit)
        categories.append({"question_id": i, "divergent": i in ids, "category": category, "ground_truth_id": hit})
    denominator = counts["A"] + counts["C"]
    precision = counts["A"] / denominator if denominator else 0.0
    recall = len(hits) / len(gt_ids) if gt_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "P": precision, "R": recall, "F1": f1,
        "categories": counts, "ground_truth_hits": sorted(hits), "per_question": categories,
    }
