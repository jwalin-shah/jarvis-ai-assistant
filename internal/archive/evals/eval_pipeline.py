#!/usr/bin/env python3  # noqa: E501
"""Baseline evaluation pipeline for the 6-category reply generation system.  # noqa: E501
  # noqa: E501
Runs the full pipeline (classifier + generation) on the gold eval dataset,  # noqa: E501
scores outputs across multiple dimensions, and produces a report.  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    uv run python evals/eval_pipeline.py                  # local checks only  # noqa: E501
    uv run python evals/eval_pipeline.py --judge           # + LLM judge scoring  # noqa: E501
    uv run python evals/eval_pipeline.py --similarity      # + BERT cosine similarity  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import json  # noqa: E501
import logging  # noqa: E501
import os  # noqa: E501
import sys  # noqa: E501
import time  # noqa: E501
from dataclasses import dataclass, field  # noqa: E402  # noqa: E501
from datetime import UTC, datetime  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501

# noqa: E501
from tqdm import tqdm  # noqa: E402  # noqa: E501

  # noqa: E501
PROJECT_ROOT = Path(__file__).parent.parent  # noqa: E501
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E501
  # noqa: E501
# Load .env  # noqa: E501
_env_path = PROJECT_ROOT / ".env"  # noqa: E501
if _env_path.exists():  # noqa: E501
    for line in _env_path.read_text().splitlines():  # noqa: E501
        line = line.strip()  # noqa: E501
        if line and not line.startswith("#") and "=" in line:  # noqa: E501
            key, _, val = line.partition("=")  # noqa: E501
            os.environ.setdefault(key.strip(), val.strip())  # noqa: E501
  # noqa: E501
ANTI_AI_PHRASES = [  # noqa: E501
    "i'd be happy to",  # noqa: E501
    "i hope this helps",  # noqa: E501
    "let me know if",  # noqa: E501
    "i understand",  # noqa: E501
    "as an ai",  # noqa: E501
    "i'm an ai",  # noqa: E501
    "certainly!",  # noqa: E501
    "of course!",  # noqa: E501
    "great question",  # noqa: E501
]  # noqa: E501
  # noqa: E501
EVAL_DATASET_PATH = PROJECT_ROOT / "evals" / "eval_dataset.jsonl"  # noqa: E501
  # noqa: E501
CATEGORIES = ["acknowledge", "closing", "question", "request", "emotion", "statement"]  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class EvalExample:  # noqa: E501
    category: str  # noqa: E501
    context: list[str]  # noqa: E501
    last_message: str  # noqa: E501
    ideal_response: str  # noqa: E501
    contact_style: str  # noqa: E501
    notes: str  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class EvalResult:  # noqa: E501
    example: EvalExample  # noqa: E501
    predicted_category: str  # noqa: E501
    generated_response: str  # noqa: E501
    latency_ms: float  # noqa: E501
    category_match: bool  # noqa: E501
    anti_ai_violations: list[str] = field(default_factory=list)  # noqa: E501
    response_length: int = 0  # noqa: E501
    route_type: str = "unknown"  # noqa: E501
    route_reason: str = ""  # noqa: E501
    similarity_score: float | None = None  # noqa: E501
    judge_score: float | None = None  # noqa: E501
    judge_reasoning: str = ""  # noqa: E501
  # noqa: E501
  # noqa: E501
def load_eval_dataset(path: Path) -> list[EvalExample]:  # noqa: E501
    """Load eval dataset from JSONL."""  # noqa: E501
    examples = []  # noqa: E501
    for line in path.read_text().splitlines():  # noqa: E501
        line = line.strip()  # noqa: E501
        if not line:  # noqa: E501
            continue  # noqa: E501
        data = json.loads(line)  # noqa: E501
        examples.append(  # noqa: E501
            EvalExample(  # noqa: E501
                category=data["category"],  # noqa: E501
                context=data["context"],  # noqa: E501
                last_message=data["last_message"],  # noqa: E501
                ideal_response=data["ideal_response"],  # noqa: E501
                contact_style=data.get("contact_style", "casual"),  # noqa: E501
                notes=data.get("notes", ""),  # noqa: E501
            )  # noqa: E501
        )  # noqa: E501
    return examples  # noqa: E501
  # noqa: E501
  # noqa: E501
def check_anti_ai(text: str) -> list[str]:  # noqa: E501
    """Check for AI-sounding phrases."""  # noqa: E501
    lower = text.lower()  # noqa: E501
    return [phrase for phrase in ANTI_AI_PHRASES if phrase in lower]  # noqa: E501
  # noqa: E501
  # noqa: E501
def ensure_realistic_thread(context: list[str], last_message: str) -> list[str]:  # noqa: E501
    """Ensure eval thread has at least two turns to avoid thin-context artifacts."""  # noqa: E501
    clean = [c for c in context if isinstance(c, str) and c.strip()]  # noqa: E501
    if len(clean) >= 2:  # noqa: E501
        return clean  # noqa: E501
    if clean:  # noqa: E501
        return ["Me: quick follow-up", clean[0]]  # noqa: E501
    return ["Me: quick follow-up", f"Them: {last_message}"]  # noqa: E501
  # noqa: E501
  # noqa: E501
def _strip_fenced_json(text: str) -> str:  # noqa: E501
    text = (text or "").strip()  # noqa: E501
    if text.startswith("```"):  # noqa: E501
        parts = text.split("```")  # noqa: E501
        if len(parts) >= 2:  # noqa: E501
            text = parts[1]  # noqa: E501
        if text.startswith("json"):  # noqa: E501
            text = text[4:]  # noqa: E501
    return text.strip()  # noqa: E501
  # noqa: E501
  # noqa: E501
def _extract_json_blob(text: str) -> str:  # noqa: E501
    """Extract the most likely JSON object/array from model output."""  # noqa: E501
    s = _strip_fenced_json(text)  # noqa: E501
    if not s:  # noqa: E501
        return s  # noqa: E501
    # Prefer array payload for batched judge; fallback to object payload.  # noqa: E501
    arr_start = s.find("[")  # noqa: E501
    arr_end = s.rfind("]")  # noqa: E501
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:  # noqa: E501
        return s[arr_start : arr_end + 1]  # noqa: E501
    obj_start = s.find("{")  # noqa: E501
    obj_end = s.rfind("}")  # noqa: E501
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:  # noqa: E501
        return s[obj_start : obj_end + 1]  # noqa: E501
    return s  # noqa: E501
  # noqa: E501
  # noqa: E501
def _judge_single_item(judge_client: object, judge_model: str, ex: EvalExample, generated: str) -> tuple[float | None, str]:  # noqa: E501
    """Judge one item and return (score, reasoning)."""  # noqa: E501
    try:  # noqa: E501
        prompt = (  # noqa: E501
            "You are an expert evaluator for a text message reply generator.\n\n"  # noqa: E501
            f"CONVERSATION CONTEXT:\n{chr(10).join(ex.context)}\n\n"  # noqa: E501
            f"LAST MESSAGE (to reply to):\n{ex.last_message}\n\n"  # noqa: E501
            f"IDEAL RESPONSE:\n{ex.ideal_response}\n\n"  # noqa: E501
            f"GENERATED REPLY:\n{generated}\n\n"  # noqa: E501
            f"CATEGORY: {ex.category}\n"  # noqa: E501
            f"NOTES: {ex.notes}\n\n"  # noqa: E501
            "Score the generated reply from 0-10. Consider:\n"  # noqa: E501
            "- Does it match the tone and intent of the ideal response?\n"  # noqa: E501
            "- Does it sound like a real person texting (not an AI)?\n"  # noqa: E501
            "- Is it appropriate for the category?\n"  # noqa: E501
            "- Is the length appropriate?\n\n"  # noqa: E501
            'Respond in JSON: {"score": <0-10>, "reasoning": "<1-2 sentences>"}'  # noqa: E501
        )  # noqa: E501
        resp = judge_client.chat.completions.create(  # noqa: E501
            model=judge_model,  # noqa: E501
            messages=[{"role": "user", "content": prompt}],  # noqa: E501
            temperature=0.0,  # noqa: E501
            max_tokens=150,  # noqa: E501
        )  # noqa: E501
        payload = json.loads(_extract_json_blob(resp.choices[0].message.content or ""))  # noqa: E501
        score = float(payload["score"])  # noqa: E501
        if score < 0:  # noqa: E501
            score = 0.0  # noqa: E501
        if score > 10:  # noqa: E501
            score = 10.0  # noqa: E501
        return score, str(payload.get("reasoning", ""))  # noqa: E501
    except Exception as e:  # noqa: E501
        return None, f"judge error: {e}"  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    import argparse  # noqa: E501
  # noqa: E501
    parser = argparse.ArgumentParser(description="JARVIS Eval Pipeline")  # noqa: E501
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge scoring")  # noqa: E501
    parser.add_argument("--similarity", action="store_true", help="Enable BERT cosine similarity")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--judge-batch-size",  # noqa: E501
        type=int,  # noqa: E501
        default=1,  # noqa: E501
        help="Judge batch size (1 = per-example judging, >1 = batched judging)",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--judge-delay-seconds",  # noqa: E501
        type=float,  # noqa: E501
        default=2.2,  # noqa: E501
        help="Delay between judge API calls/batches to reduce rate-limit errors",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--force-model-load",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Set JARVIS_FORCE_MODEL_LOAD=1 to bypass memory-pressure load guard (risky)",  # noqa: E501
    )  # noqa: E501
    args = parser.parse_args()  # noqa: E501
    if args.force_model_load:  # noqa: E501
        os.environ["JARVIS_FORCE_MODEL_LOAD"] = "1"  # noqa: E501
  # noqa: E501
    # Setup logging  # noqa: E501
    log_path = PROJECT_ROOT / "results" / "eval_pipeline.log"  # noqa: E501
    log_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
    logging.basicConfig(  # noqa: E501
        level=logging.INFO,  # noqa: E501
        format="%(asctime)s - %(levelname)s - %(message)s",  # noqa: E501
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],  # noqa: E501
    )  # noqa: E501
    logging.getLogger(__name__)  # noqa: E501
  # noqa: E501
    # Load dataset  # noqa: E501
    if not EVAL_DATASET_PATH.exists():  # noqa: E501
        print(f"ERROR: Eval dataset not found at {EVAL_DATASET_PATH}", flush=True)  # noqa: E501
        return 1  # noqa: E501
  # noqa: E501
    examples = load_eval_dataset(EVAL_DATASET_PATH)  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
    print("JARVIS EVAL PIPELINE - Baseline Measurement", flush=True)  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
    print(f"Examples:    {len(examples)}", flush=True)  # noqa: E501
    print(f"Categories:  {', '.join(CATEGORIES)}", flush=True)  # noqa: E501
    print(f"Judge:       {'enabled' if args.judge else 'disabled (use --judge)'}", flush=True)  # noqa: E501
    print(  # noqa: E501
        f"Similarity:  {'enabled' if args.similarity else 'disabled (use --similarity)'}",  # noqa: E501
        flush=True,  # noqa: E501
    )  # noqa: E501
    print(flush=True)  # noqa: E501
  # noqa: E501
    # Initialize components  # noqa: E501
    print("Loading classifier...", flush=True)  # noqa: E501
    from jarvis.classifiers.category_classifier import classify_category  # noqa: E501
    from jarvis.classifiers.response_mobilization import classify_response_pressure  # noqa: E501
  # noqa: E501
    # Initialize reply service for generation  # noqa: E501
    print("Loading reply service...", flush=True)  # noqa: E501
    from jarvis.contracts.pipeline import MessageContext  # noqa: E501
    from jarvis.reply_service import ReplyService  # noqa: E501
  # noqa: E501
    reply_service = ReplyService()  # noqa: E501
  # noqa: E501
    # Optional: BERT embedder for similarity  # noqa: E501
    embedder = None  # noqa: E501
    if args.similarity:  # noqa: E501
        print("Loading embedder for similarity...", flush=True)  # noqa: E501
        from jarvis.embedding_adapter import get_embedder  # noqa: E501
  # noqa: E501
        embedder = get_embedder()  # noqa: E501
  # noqa: E501
    # Optional: judge  # noqa: E501
    judge_client = None  # noqa: E501
    if args.judge:  # noqa: E501
        from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E501
  # noqa: E501
        judge_client = get_judge_client()  # noqa: E501
        if judge_client is None:  # noqa: E501
            print("WARNING: Judge API key not set, skipping judge", flush=True)  # noqa: E501
        else:  # noqa: E501
            print(f"Judge ready: {JUDGE_MODEL}", flush=True)  # noqa: E501
  # noqa: E501
    print(flush=True)  # noqa: E501
    print("-" * 70, flush=True)  # noqa: E501
  # noqa: E501
    results: list[EvalResult] = []  # noqa: E501
    total_start = time.perf_counter()  # noqa: E501
  # noqa: E501
    for i, ex in enumerate(tqdm(examples, desc="Evaluating"), 1):  # noqa: E501
        print(f"\n[{i:2d}/{len(examples)}] [{ex.category}] {ex.last_message[:50]}...", flush=True)  # noqa: E501
  # noqa: E501
        gen_start = time.perf_counter()  # noqa: E501
        eval_thread = ensure_realistic_thread(ex.context, ex.last_message)  # noqa: E501
  # noqa: E501
        # 1. Classify category  # noqa: E501
        mobilization = classify_response_pressure(ex.last_message)  # noqa: E501
        category_result = classify_category(  # noqa: E501
            ex.last_message,  # noqa: E501
            context=eval_thread,  # noqa: E501
            mobilization=mobilization,  # noqa: E501
        )  # noqa: E501
        predicted_category = category_result.category  # noqa: E501
  # noqa: E501
        # 2. Generate reply via full pipeline (empty search_results to skip RAG)  # noqa: E501
        route_type = "unknown"  # noqa: E501
        route_reason = ""  # noqa: E501
        try:  # noqa: E501
            context = MessageContext(  # noqa: E501
                chat_id="iMessage;-;+15555550123",  # noqa: E501
                message_text=ex.last_message,  # noqa: E501
                is_from_me=False,  # noqa: E501
                timestamp=datetime.now(UTC),  # noqa: E501
                metadata={"thread": eval_thread, "contact_name": "John"},  # noqa: E501
            )  # noqa: E501
            reply_result = reply_service.generate_reply(  # noqa: E501
                context=context,  # noqa: E501
                thread=eval_thread,  # noqa: E501
                search_results=[],  # noqa: E501
            )  # noqa: E501
            generated = reply_result.response  # noqa: E501
            route_type = str(reply_result.metadata.get("type", "unknown"))  # noqa: E501
            route_reason = str(reply_result.metadata.get("reason", ""))  # noqa: E501
        except Exception as e:  # noqa: E501
            generated = f"[ERROR: {e}]"  # noqa: E501
            route_type = "error"  # noqa: E501
            route_reason = str(e)  # noqa: E501
  # noqa: E501
        latency_ms = (time.perf_counter() - gen_start) * 1000  # noqa: E501
  # noqa: E501
        # 3. Score  # noqa: E501
        category_match = predicted_category == ex.category  # noqa: E501
        anti_ai = check_anti_ai(generated)  # noqa: E501
  # noqa: E501
        # Similarity scoring  # noqa: E501
        sim_score = None  # noqa: E501
        if embedder and generated and not generated.startswith("[ERROR"):  # noqa: E501
            try:  # noqa: E501
                import numpy as np  # noqa: E501
  # noqa: E501
                emb_gen = embedder.encode([generated])[0]  # noqa: E501
                emb_ideal = embedder.encode([ex.ideal_response])[0]  # noqa: E501
                sim_score = float(  # noqa: E501
                    np.dot(emb_gen, emb_ideal)  # noqa: E501
                    / (np.linalg.norm(emb_gen) * np.linalg.norm(emb_ideal))  # noqa: E501
                )  # noqa: E501
            except Exception:  # noqa: E501
                pass  # noqa: E501
  # noqa: E501
        # Judge scoring  # noqa: E501
        j_score = None  # noqa: E501
        j_reasoning = ""  # noqa: E501
        if (  # noqa: E501
            judge_client  # noqa: E501
            and args.judge_batch_size <= 1  # noqa: E501
            and generated  # noqa: E501
            and not generated.startswith("[ERROR")  # noqa: E501
        ):  # noqa: E501
            j_score, j_reasoning = _judge_single_item(judge_client, JUDGE_MODEL, ex, generated)  # noqa: E501
            if args.judge_delay_seconds > 0:  # noqa: E501
                time.sleep(args.judge_delay_seconds)  # noqa: E501
  # noqa: E501
        result = EvalResult(  # noqa: E501
            example=ex,  # noqa: E501
            predicted_category=predicted_category,  # noqa: E501
            generated_response=generated,  # noqa: E501
            latency_ms=latency_ms,  # noqa: E501
            category_match=category_match,  # noqa: E501
            anti_ai_violations=anti_ai,  # noqa: E501
            response_length=len(generated),  # noqa: E501
            route_type=route_type,  # noqa: E501
            route_reason=route_reason,  # noqa: E501
            similarity_score=sim_score,  # noqa: E501
            judge_score=j_score,  # noqa: E501
            judge_reasoning=j_reasoning,  # noqa: E501
        )  # noqa: E501
        results.append(result)  # noqa: E501
  # noqa: E501
        # Print per-example  # noqa: E501
        cat_status = "OK" if category_match else f"MISS (got {predicted_category})"  # noqa: E501
        ai_status = f"AI:{len(anti_ai)}" if anti_ai else "clean"  # noqa: E501
        sim_str = f"sim={sim_score:.2f}" if sim_score is not None else ""  # noqa: E501
        judge_str = f"judge={j_score:.0f}/10" if j_score is not None else ""  # noqa: E501
        print(f'  Response: "{generated[:60]}"', flush=True)  # noqa: E501
        print(  # noqa: E501
            f"  Cat: {cat_status} | {ai_status} | {latency_ms:.0f}ms {sim_str} {judge_str}",  # noqa: E501
            flush=True,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    # Optional batched judge pass (reduces API requests significantly)  # noqa: E501
    if judge_client and args.judge_batch_size > 1:  # noqa: E501
        judgeable_idx = [  # noqa: E501
            i  # noqa: E501
            for i, r in enumerate(results)  # noqa: E501
            if r.generated_response and not r.generated_response.startswith("[ERROR")  # noqa: E501
        ]  # noqa: E501
        if judgeable_idx:  # noqa: E501
            print(  # noqa: E501
                f"\nRunning batched judge: {len(judgeable_idx)} items, "  # noqa: E501
                f"batch_size={args.judge_batch_size}",  # noqa: E501
                flush=True,  # noqa: E501
            )  # noqa: E501
            for start in range(0, len(judgeable_idx), args.judge_batch_size):  # noqa: E501
                chunk = judgeable_idx[start : start + args.judge_batch_size]  # noqa: E501
                batch_prompt = (  # noqa: E501
                    "You are an expert evaluator for text message replies.\n"  # noqa: E501
                    "For each item, score generated reply 0-10 and provide brief reasoning.\n"  # noqa: E501
                    "Return ONLY JSON array with objects: "  # noqa: E501
                    '{"index": <1-based item index in this batch>, "score": <0-10>, '  # noqa: E501
                    '"reasoning": "<1-2 sentences>"}.\n\n'  # noqa: E501
                )  # noqa: E501
                for pos, idx in enumerate(chunk, 1):  # noqa: E501
                    r = results[idx]  # noqa: E501
                    ex = r.example  # noqa: E501
                    batch_prompt += (  # noqa: E501
                        f"ITEM {pos}\n"  # noqa: E501
                        f"CATEGORY: {ex.category}\n"  # noqa: E501
                        f"CONTEXT:\n{chr(10).join(ex.context)}\n"  # noqa: E501
                        f"LAST MESSAGE:\n{ex.last_message}\n"  # noqa: E501
                        f"IDEAL RESPONSE:\n{ex.ideal_response}\n"  # noqa: E501
                        f"GENERATED REPLY:\n{r.generated_response}\n\n"  # noqa: E501
                    )  # noqa: E501
                try:  # noqa: E501
                    resp = judge_client.chat.completions.create(  # noqa: E501
                        model=JUDGE_MODEL,  # noqa: E501
                        messages=[{"role": "user", "content": batch_prompt}],  # noqa: E501
                        temperature=0.0,  # noqa: E501
                        max_tokens=600,  # noqa: E501
                    )  # noqa: E501
                    text = _extract_json_blob(resp.choices[0].message.content or "")  # noqa: E501
                    payload = json.loads(text)  # noqa: E501
                    assigned: set[int] = set()  # noqa: E501
                    payload_items = payload  # noqa: E501
                    if isinstance(payload, dict):  # noqa: E501
                        payload_items = payload.get("items", [])  # noqa: E501
                    if isinstance(payload_items, list):  # noqa: E501
                        for i, item in enumerate(payload_items, 1):  # noqa: E501
                            try:  # noqa: E501
                                if isinstance(item, dict):  # noqa: E501
                                    local_idx = int(item.get("index", i))  # noqa: E501
                                    score = float(item.get("score"))  # noqa: E501
                                    reasoning = str(item.get("reasoning", ""))  # noqa: E501
                                elif isinstance(item, list | tuple) and len(item) >= 2:  # noqa: E501
                                    local_idx = i  # noqa: E501
                                    score = float(item[0])  # noqa: E501
                                    reasoning = str(item[1])  # noqa: E501
                                else:  # noqa: E501
                                    continue  # noqa: E501
                                if 1 <= local_idx <= len(chunk):  # noqa: E501
                                    if score < 0:  # noqa: E501
                                        score = 0.0  # noqa: E501
                                    if score > 10:  # noqa: E501
                                        score = 10.0  # noqa: E501
                                    target = results[chunk[local_idx - 1]]  # noqa: E501
                                    target.judge_score = score  # noqa: E501
                                    target.judge_reasoning = reasoning  # noqa: E501
                                    assigned.add(local_idx)  # noqa: E501
                            except Exception:  # noqa: E501
                                continue  # noqa: E501
                    for local_idx, idx in enumerate(chunk, 1):  # noqa: E501
                        if local_idx in assigned:  # noqa: E501
                            continue  # noqa: E501
                        score, reasoning = _judge_single_item(  # noqa: E501
                            judge_client,  # noqa: E501
                            JUDGE_MODEL,  # noqa: E501
                            results[idx].example,  # noqa: E501
                            results[idx].generated_response,  # noqa: E501
                        )  # noqa: E501
                        results[idx].judge_score = score  # noqa: E501
                        results[idx].judge_reasoning = reasoning  # noqa: E501
                        if args.judge_delay_seconds > 0:  # noqa: E501
                            time.sleep(min(args.judge_delay_seconds, 0.6))  # noqa: E501
                except Exception as e:  # noqa: E501
                    for idx in chunk:  # noqa: E501
                        score, reasoning = _judge_single_item(  # noqa: E501
                            judge_client,  # noqa: E501
                            JUDGE_MODEL,  # noqa: E501
                            results[idx].example,  # noqa: E501
                            results[idx].generated_response,  # noqa: E501
                        )  # noqa: E501
                        results[idx].judge_score = score  # noqa: E501
                        results[idx].judge_reasoning = (  # noqa: E501
                            f"batch_fail_then_single: {reasoning}; batch_error={e}"  # noqa: E501
                        )  # noqa: E501
                        if args.judge_delay_seconds > 0:  # noqa: E501
                            time.sleep(min(args.judge_delay_seconds, 0.6))  # noqa: E501
                if args.judge_delay_seconds > 0 and (start + args.judge_batch_size) < len(  # noqa: E501
                    judgeable_idx  # noqa: E501
                ):  # noqa: E501
                    time.sleep(args.judge_delay_seconds)  # noqa: E501
  # noqa: E501
    total_ms = (time.perf_counter() - total_start) * 1000  # noqa: E501
  # noqa: E501
    # =========================================================================  # noqa: E501
    # Summary Report  # noqa: E501
    # =========================================================================  # noqa: E501
    print(flush=True)  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
    print("SUMMARY", flush=True)  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
  # noqa: E501
    n = len(results)  # noqa: E501
    if n == 0:  # noqa: E501
        print("No results to summarize.", flush=True)  # noqa: E501
        return 0  # noqa: E501
  # noqa: E501
    cat_matches = sum(1 for r in results if r.category_match)  # noqa: E501
    ai_clean = sum(1 for r in results if not r.anti_ai_violations)  # noqa: E501
    latencies = [r.latency_ms for r in results]  # noqa: E501
    avg_lat = sum(latencies) / n  # noqa: E501
    sorted_lat = sorted(latencies)  # noqa: E501
    p50 = sorted_lat[n // 2]  # noqa: E501
    p95 = sorted_lat[min(int(n * 0.95), n - 1)]  # noqa: E501
  # noqa: E501
    print(f"Category accuracy:  {cat_matches}/{n} ({cat_matches / n * 100:.0f}%)", flush=True)  # noqa: E501
    print(f"Anti-AI clean:      {ai_clean}/{n} ({ai_clean / n * 100:.0f}%)", flush=True)  # noqa: E501
    print(f"Total time:         {total_ms:.0f}ms", flush=True)  # noqa: E501
    print(f"Avg latency:        {avg_lat:.0f}ms", flush=True)  # noqa: E501
    print(f"P50/P95 latency:    {p50:.0f}ms / {p95:.0f}ms", flush=True)  # noqa: E501
  # noqa: E501
    # Route-path summary  # noqa: E501
    print(flush=True)  # noqa: E501
    print("ROUTE PATHS", flush=True)  # noqa: E501
    print("-" * 70, flush=True)  # noqa: E501
    route_counts: dict[str, int] = {}  # noqa: E501
    route_empty_counts: dict[str, int] = {}  # noqa: E501
    for r in results:  # noqa: E501
        key = f"{r.route_type}:{r.route_reason}" if r.route_reason else r.route_type  # noqa: E501
        route_counts[key] = route_counts.get(key, 0) + 1  # noqa: E501
        if not (r.generated_response or "").strip():  # noqa: E501
            route_empty_counts[key] = route_empty_counts.get(key, 0) + 1  # noqa: E501
    for key, count in sorted(route_counts.items(), key=lambda kv: kv[1], reverse=True):  # noqa: E501
        empty = route_empty_counts.get(key, 0)  # noqa: E501
        print(f"  {key:<35} {count:>2d} (empty={empty})", flush=True)  # noqa: E501
  # noqa: E501
    # Similarity summary  # noqa: E501
    scored_sim = [r for r in results if r.similarity_score is not None]  # noqa: E501
    if scored_sim:  # noqa: E501
        sims = [r.similarity_score for r in scored_sim]  # noqa: E501
        print(f"Avg similarity:     {sum(sims) / len(sims):.3f}", flush=True)  # noqa: E501
  # noqa: E501
    # Judge summary  # noqa: E501
    scored_judge = [r for r in results if r.judge_score is not None and r.judge_score >= 0]  # noqa: E501
    if scored_judge:  # noqa: E501
        scores = [r.judge_score for r in scored_judge]  # noqa: E501
        avg_j = sum(scores) / len(scores)  # noqa: E501
        pass_7 = sum(1 for s in scores if s >= 7)  # noqa: E501
        print(f"Judge avg:          {avg_j:.1f}/10", flush=True)  # noqa: E501
        print(  # noqa: E501
            f"Judge pass (>=7):   {pass_7}/{len(scores)} ({pass_7 / len(scores) * 100:.0f}%)",  # noqa: E501
            flush=True,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    # Per-category breakdown  # noqa: E501
    print(flush=True)  # noqa: E501
    print("PER-CATEGORY BREAKDOWN", flush=True)  # noqa: E501
    print("-" * 70, flush=True)  # noqa: E501
    for cat in CATEGORIES:  # noqa: E501
        cat_results = [r for r in results if r.example.category == cat]  # noqa: E501
        if not cat_results:  # noqa: E501
            continue  # noqa: E501
        cat_correct = sum(1 for r in cat_results if r.category_match)  # noqa: E501
        cat_clean = sum(1 for r in cat_results if not r.anti_ai_violations)  # noqa: E501
        parts = [  # noqa: E501
            f"classify={cat_correct}/{len(cat_results)}",  # noqa: E501
            f"clean={cat_clean}/{len(cat_results)}",  # noqa: E501
        ]  # noqa: E501
        cat_sim = [r for r in cat_results if r.similarity_score is not None]  # noqa: E501
        if cat_sim:  # noqa: E501
            avg_s = sum(r.similarity_score for r in cat_sim) / len(cat_sim)  # noqa: E501
            parts.append(f"sim={avg_s:.2f}")  # noqa: E501
        cat_j = [r for r in cat_results if r.judge_score is not None and r.judge_score >= 0]  # noqa: E501
        if cat_j:  # noqa: E501
            avg_jj = sum(r.judge_score for r in cat_j) / len(cat_j)  # noqa: E501
            parts.append(f"judge={avg_jj:.1f}")  # noqa: E501
        print(f"  {cat:15s}  {' | '.join(parts)}", flush=True)  # noqa: E501
  # noqa: E501
    # Category misclassifications  # noqa: E501
    misses = [r for r in results if not r.category_match]  # noqa: E501
    if misses:  # noqa: E501
        print(flush=True)  # noqa: E501
        print("CATEGORY MISCLASSIFICATIONS", flush=True)  # noqa: E501
        print("-" * 70, flush=True)  # noqa: E501
        for r in misses:  # noqa: E501
            print(  # noqa: E501
                f"  [{r.example.category} -> {r.predicted_category}] {r.example.last_message[:50]}",  # noqa: E501
                flush=True,  # noqa: E501
            )  # noqa: E501
  # noqa: E501
    # Anti-AI violations  # noqa: E501
    violations = [r for r in results if r.anti_ai_violations]  # noqa: E501
    if violations:  # noqa: E501
        print(flush=True)  # noqa: E501
        print("ANTI-AI VIOLATIONS", flush=True)  # noqa: E501
        print("-" * 70, flush=True)  # noqa: E501
        for r in violations:  # noqa: E501
            print(f'  "{r.generated_response[:60]}" -> {r.anti_ai_violations}', flush=True)  # noqa: E501
  # noqa: E501
    # Worst judge scores  # noqa: E501
    if scored_judge:  # noqa: E501
        low = sorted(scored_judge, key=lambda r: r.judge_score)[:5]  # noqa: E501
        if low and low[0].judge_score < 7:  # noqa: E501
            print(flush=True)  # noqa: E501
            print("LOWEST JUDGE SCORES", flush=True)  # noqa: E501
            print("-" * 70, flush=True)  # noqa: E501
            for r in low:  # noqa: E501
                if r.judge_score < 7:  # noqa: E501
                    print(  # noqa: E501
                        f"  [{r.example.category}] {r.judge_score:.0f}/10 - "  # noqa: E501
                        f'"{r.generated_response[:50]}" - {r.judge_reasoning}',  # noqa: E501
                        flush=True,  # noqa: E501
                    )  # noqa: E501
  # noqa: E501
    # Save results  # noqa: E501
    output_path = PROJECT_ROOT / "results" / "eval_pipeline_baseline.json"  # noqa: E501
    output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
    output_data = {  # noqa: E501
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E501
        "total_examples": n,  # noqa: E501
        "category_accuracy": round(cat_matches / n, 4) if n else 0,  # noqa: E501
        "anti_ai_clean_rate": round(ai_clean / n, 4) if n else 0,  # noqa: E501
        "avg_similarity": round(sum(sims) / len(sims), 4) if scored_sim else None,  # noqa: E501
        "judge_avg": round(avg_j, 2) if scored_judge else None,  # noqa: E501
        "latency": {  # noqa: E501
            "avg_ms": round(avg_lat, 1),  # noqa: E501
            "p50_ms": round(p50, 1),  # noqa: E501
            "p95_ms": round(p95, 1),  # noqa: E501
            "total_ms": round(total_ms, 1),  # noqa: E501
        },  # noqa: E501
        "per_category": {},  # noqa: E501
        "results": [  # noqa: E501
            {  # noqa: E501
                "category_expected": r.example.category,  # noqa: E501
                "category_predicted": r.predicted_category,  # noqa: E501
                "last_message": r.example.last_message,  # noqa: E501
                "ideal_response": r.example.ideal_response,  # noqa: E501
                "generated_response": r.generated_response,  # noqa: E501
                "category_match": r.category_match,  # noqa: E501
                "anti_ai_violations": r.anti_ai_violations,  # noqa: E501
                "similarity_score": r.similarity_score,  # noqa: E501
                "judge_score": r.judge_score,  # noqa: E501
                "judge_reasoning": r.judge_reasoning,  # noqa: E501
                "latency_ms": round(r.latency_ms, 1),  # noqa: E501
                "route_type": r.route_type,  # noqa: E501
                "route_reason": r.route_reason,  # noqa: E501
            }  # noqa: E501
            for r in results  # noqa: E501
        ],  # noqa: E501
    }  # noqa: E501
  # noqa: E501
    # Per-category stats  # noqa: E501
    for cat in CATEGORIES:  # noqa: E501
        cat_results = [r for r in results if r.example.category == cat]  # noqa: E501
        if not cat_results:  # noqa: E501
            continue  # noqa: E501
        output_data["per_category"][cat] = {  # noqa: E501
            "count": len(cat_results),  # noqa: E501
            "classify_accuracy": round(  # noqa: E501
                sum(1 for r in cat_results if r.category_match) / len(cat_results), 4  # noqa: E501
            ),  # noqa: E501
            "anti_ai_clean_rate": round(  # noqa: E501
                sum(1 for r in cat_results if not r.anti_ai_violations) / len(cat_results), 4  # noqa: E501
            ),  # noqa: E501
        }  # noqa: E501
  # noqa: E501
    output_path.write_text(json.dumps(output_data, indent=2))  # noqa: E501
    print(f"\nResults saved to: {output_path}", flush=True)  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
  # noqa: E501
    return 0  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
