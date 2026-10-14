"""Evaluate a prober with IoA-Suite MCQ probing."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

from openai import OpenAI

import ioa

ROOT = Path(__file__).resolve().parent
ROLES = ("prober", "simulator", "validity_filter")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_endpoints(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    clients, settings = {}, {}
    for role in ROLES:
        item = config[role]
        prefix = role.upper()
        base_url = os.getenv(f"{prefix}_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv(f"{prefix}_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not base_url or not api_key:
            raise ValueError(f"Set OPENAI_BASE_URL and OPENAI_API_KEY (or {prefix}_BASE_URL / {prefix}_API_KEY).")
        url = urlsplit(base_url)
        if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError("Base URLs must be HTTP(S) endpoints without credentials, query strings or fragments.")
        if not isinstance(item.get("model"), str) or not item["model"].strip():
            raise ValueError(f"Set the {role} model in the config.")
        options = item.get("request_options", {})
        allowed = {"extra_body", "reasoning_effort", "max_tokens", "max_completion_tokens", "top_p"}
        if not isinstance(options, dict) or set(options) - allowed:
            raise ValueError(f"Unsupported request_options for {role}; allowed: {sorted(allowed)}.")
        extra = options.get("extra_body", {})
        if not isinstance(extra, dict) or set(extra) & {"model", "messages", "temperature", "stream"}:
            raise ValueError("extra_body cannot replace the model, messages, temperature or stream setting.")
        temperature = item.get("temperature", 0.7 if role == "prober" else 0.2)
        if temperature is not None and (type(temperature) not in (int, float) or not 0 <= temperature <= 2):
            raise ValueError(f"Invalid temperature for {role}.")
        settings[role] = {"model": item["model"], "base_url": base_url.rstrip("/"), "temperature": temperature, "request_options": options}
        clients[role] = OpenAI(api_key=api_key, base_url=base_url, timeout=180.0, max_retries=2)
    return clients, settings


def call(clients, settings, role, messages, calls, participant_id=None):
    system, user = messages
    entry = {"role": role, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if participant_id is not None:
        entry["participant_id"] = participant_id
    calls.append(entry)
    item = settings[role]
    arguments = {"model": item["model"], "messages": entry["messages"], **item["request_options"]}
    if item["temperature"] is not None:
        arguments["temperature"] = item["temperature"]
    response = clients[role].chat.completions.create(**arguments)
    if not response.choices:
        raise ValueError("The API returned no choices.")
    choice = response.choices[0]
    entry["response"] = {
        "id": response.id, "model": response.model,
        "content": choice.message.content,
        "reasoning_content": getattr(choice.message, "reasoning_content", None),
        "finish_reason": choice.finish_reason,
        "usage": response.usage.model_dump() if response.usage else None,
    }
    if choice.finish_reason != "stop":
        raise ValueError(f"Incomplete response (finish_reason={choice.finish_reason}).")
    return ioa.parse_response(choice.message.content)


def evaluate_case(case, clients, settings, record):
    participants = case["profiles"]["members"]
    if len(participants) != 2 or len({p["id"] for p in participants}) != 2:
        raise ValueError("Each IoA-Suite dialogue must have two distinct participants.")
    ground_truth = case["profiles"]["misalignment_points"]
    gt_ids = {p["id"] for p in ground_truth}
    if not gt_ids or len(gt_ids) != len(ground_truth):
        raise ValueError("Ground truth IDs must be unique and non-empty.")
    record["prober"] = call(clients, settings, "prober", ioa.prober_messages(case), record["calls"])
    questions = ioa.validate_questions(record["prober"])
    answers = []
    record["participant_answers"] = {}
    for participant in participants:
        result = call(
            clients, settings, "simulator", ioa.simulator_messages(case, participant, questions),
            record["calls"], participant["id"],
        ) if questions else {"answers": []}
        answer = ioa.validate_answers(result, questions)
        record["participant_answers"][participant["id"]] = answer
        answers.append(answer)
    ids = ioa.divergent_ids(questions, answers)
    record["validity_filter"] = call(
        clients, settings, "validity_filter", ioa.validity_filter_messages(case, questions, answers), record["calls"],
    ) if ids else {"evaluations": []}
    ioa.validate_evaluations(record["validity_filter"], ids, gt_ids)
    record["metrics"] = ioa.score(questions, answers, record["validity_filter"], ground_truth)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--data", type=Path, default=Path("data/test"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive.")
    paths = sorted(args.data.rglob("*.json")) if args.data.is_dir() else []
    if not paths:
        parser.error("--data must be a directory containing IoA-Suite case JSON files.")
    if args.limit:
        paths = paths[:args.limit]
    try:
        clients, settings = load_endpoints(args.config)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    source_paths = [ROOT / "ioa.py", ROOT / "evaluate.py", *sorted(ioa.PROMPTS.glob("*.txt"))]
    protocol_hash = digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    run = {"endpoints": settings, "protocol_sha256": protocol_hash}
    run_id = digest(run)
    manifest_path = args.output / "run.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != run:
        parser.error("The output directory belongs to different settings or code; choose a new --output directory.")
    jobs = []
    for path in paths:
        relative = path.relative_to(args.data)
        data = path.read_bytes()
        fingerprint = digest({"run": run_id, "case_sha256": hashlib.sha256(data).hexdigest()})
        destination = args.output / "records" / relative
        if destination.exists():
            previous = json.loads(destination.read_text())
            if previous.get("fingerprint") != fingerprint:
                parser.error(f"The data for {relative.as_posix()} changed; choose a new --output directory.")
        jobs.append((relative.as_posix(), json.loads(data), fingerprint, destination))
    save_json(manifest_path, run)
    (args.output / "summary.json").unlink(missing_ok=True)

    def run_one(job):
        case_id, case, fingerprint, destination = job
        if destination.exists():
            existing = json.loads(destination.read_text())
            if existing.get("status") == "ok":
                return existing
        record = {"case": case_id, "fingerprint": fingerprint, "started_at": datetime.now(timezone.utc).isoformat(), "calls": []}
        try:
            evaluate_case(case, clients, settings, record)
            record["status"] = "ok"
        except Exception as error:
            record["status"] = "error"
            record["error"] = {"type": type(error).__name__, "stage": record["calls"][-1]["role"] if record["calls"] else "input"}
            if isinstance(error, (ValueError, KeyError, TypeError)):
                record["error"]["message"] = str(error)
            if hasattr(error, "status_code"):
                record["error"]["http_status"] = error.status_code
        save_json(destination, record)
        return record

    records = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for record in executor.map(run_one, jobs):
                records.append(record)
                print(f"[{len(records)}/{len(jobs)}] {record['case']}: {record['status']}", flush=True)
    finally:
        for client in clients.values():
            client.close()
    failed = [r["case"] for r in records if r["status"] != "ok"]
    summary = {"cases": len(records), "completed": len(records) - len(failed), "failed": failed, "aggregation": "mean over per-dialogue scores", "P": None, "R": None, "F1": None}
    if not failed:
        for metric in ("P", "R", "F1"):
            summary[metric] = sum(record["metrics"][metric] for record in records) / len(records)
    save_json(args.output / "summary.json", summary)
    if failed:
        print("Evaluation incomplete. Inspect error records and rerun the same command to resume.", file=sys.stderr)
        return 1
    print("  ".join(f"{metric}={100 * summary[metric]:.1f}%" for metric in ("P", "R", "F1")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
