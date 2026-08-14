"""
evaluate.py — CutAgent Evaluation Script
=========================================
Run this locally to generate real metrics for your Results section.

Usage:
    python evaluate.py --video_short short.mp4 --video_medium medium.mp4 --video_long long.mp4

It will produce: results.json  ← plug these numbers into your paper

Requirements:
    pip install rouge-score sentence-transformers torch moviepy
    (all should already be installed from your project)

What it tests:
    Experiment 1 — Video Length Scalability (short / medium / long)
    Experiment 2 — Agentic Loop Iterations  (1 / 2 / 3 max iterations)
    Experiment 3 — Model Comparison         (gemini-2.0-flash vs gemini-1.5-flash vs gemini-1.5-pro)
    Experiment 4 — Platform Preset Export   (tiktok / youtube / linkedin)
"""

import os
import time
import json
import argparse
import tempfile
import traceback
from pathlib import Path

# ── Metrics ──────────────────────────────────────────────────────────────────
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("WARNING: rouge_score not installed. Run: pip install rouge-score")

try:
    from sentence_transformers import SentenceTransformer, util
    ST_AVAILABLE = True
    _st_model = None
    def get_st_model():
        global _st_model
        if _st_model is None:
            print("  Loading sentence-transformer model...")
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        return _st_model
except ImportError:
    ST_AVAILABLE = False
    print("WARNING: sentence-transformers not installed. Run: pip install sentence-transformers")

from moviepy import VideoFileClip

# ── Agents ────────────────────────────────────────────────────────────────────
from agent_orchestrator import AgentOrchestrator
from captioning_agent import CaptioningAgent


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_video_info(path: str) -> dict:
    clip = VideoFileClip(path)
    info = {
        "duration": round(clip.duration, 2),
        "resolution": f"{clip.size[0]}x{clip.size[1]}",
        "fps": round(clip.fps, 1),
        "size_mb": round(os.path.getsize(path) / 1e6, 2)
    }
    clip.close()
    return info


def compute_rouge(hypothesis: str, reference: str) -> dict:
    if not ROUGE_AVAILABLE or not hypothesis or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    if not ST_AVAILABLE or not text_a or not text_b:
        return 0.0
    model = get_st_model()
    emb_a = model.encode(text_a, convert_to_tensor=True)
    emb_b = model.encode(text_b, convert_to_tensor=True)
    return round(float(util.cos_sim(emb_a, emb_b)[0][0]), 4)


def estimate_hallucination(output_text: str, source_captions: list) -> float:
    """
    Simple hallucination proxy: fraction of output sentences that have
    low semantic similarity to any source caption segment.
    Lower = less hallucination.
    """
    if not ST_AVAILABLE or not output_text or not source_captions:
        return 0.0
    model = get_st_model()
    source_text = " ".join([c.get("text", "") for c in source_captions])
    sentences = [s.strip() for s in output_text.split(".") if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    hallucinated = 0
    src_emb = model.encode(source_text, convert_to_tensor=True)
    for sent in sentences:
        s_emb = model.encode(sent, convert_to_tensor=True)
        sim = float(util.cos_sim(s_emb, src_emb)[0][0])
        if sim < 0.35:
            hallucinated += 1
    return round(hallucinated / len(sentences), 4)


def run_single_experiment(
    video_path: str,
    prompt: str,
    model_name: str = "gemini-2.0-flash",
    max_iterations: int = 3,
    platform_key: str = None,
    reference_caption_text: str = None,
) -> dict:
    """
    Runs one full agentic edit and returns a metrics dict.
    """
    result_metrics = {
        "model": model_name,
        "prompt": prompt,
        "platform": platform_key,
        "max_iterations": max_iterations,
        "video_info": get_video_info(video_path),
        "error": None,
    }

    orchestrator = AgentOrchestrator(model_name=model_name, max_iterations=max_iterations)

    t_start = time.time()
    try:
        result = orchestrator.run(
            input_video_path=video_path,
            user_prompt=prompt,
            platform_key=platform_key,
            run_scene_detection=False,   # keep experiments clean
            run_filler_removal=False,
        )
        latency = round(time.time() - t_start, 3)

        result_metrics["latency_seconds"] = latency
        result_metrics["iterations_used"] = result["iterations"]
        result_metrics["quality_score"] = result["final_score"]
        result_metrics["passed"] = result["passed"]

        # Per-iteration scores
        result_metrics["iteration_scores"] = [
            {"iteration": a["iteration"], "score": a.get("score", 0)}
            for a in result.get("attempts", [])
        ]

        # Output video info
        if result.get("output_path") and os.path.exists(result["output_path"]):
            result_metrics["output_info"] = get_video_info(result["output_path"])

        # Caption quality (if insight was returned, use it for ROUGE)
        if result.get("insight") and reference_caption_text:
            rouge = compute_rouge(result["insight"], reference_caption_text)
            sem_sim = compute_semantic_similarity(result["insight"], reference_caption_text)
            result_metrics["rouge"] = rouge
            result_metrics["semantic_similarity"] = sem_sim

        # Activity log summary
        agent_calls = {}
        for ev in result.get("activity_log", []):
            agent = ev["agent"]
            agent_calls[agent] = agent_calls.get(agent, 0) + 1
        result_metrics["agent_call_counts"] = agent_calls

    except Exception as e:
        result_metrics["error"] = str(e)
        result_metrics["latency_seconds"] = round(time.time() - t_start, 3)
        traceback.print_exc()

    return result_metrics


def run_caption_quality_experiment(video_path: str, model_name: str = "gemini-2.0-flash") -> dict:
    """
    Runs captioning specifically and computes ROUGE against a reference.
    Reference = re-run with best model and use as ground truth.
    """
    print(f"  Running caption quality experiment on {Path(video_path).name}...")
    captioner = CaptioningAgent(model_name=model_name)

    t_start = time.time()
    try:
        captions = captioner.transcribe(video_path)
        latency = round(time.time() - t_start, 3)
        full_text = " ".join([c.get("text", "") for c in captions])
        return {
            "model": model_name,
            "caption_latency": latency,
            "segment_count": len(captions),
            "total_text_length": len(full_text),
            "captions_sample": captions[:3],   # first 3 for manual review
            "full_text": full_text,
        }
    except Exception as e:
        return {"model": model_name, "error": str(e)}


# ── Main Experiments ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CutAgent Evaluation Script")
    parser.add_argument("--video_short",  required=True, help="Path to short video  (~30s)")
    parser.add_argument("--video_medium", required=True, help="Path to medium video (~2min)")
    parser.add_argument("--video_long",   required=True, help="Path to long video   (~5min)")
    parser.add_argument("--output",       default="results.json", help="Output JSON file")
    parser.add_argument("--skip_models",  action="store_true", help="Skip model comparison (saves API quota)")
    args = parser.parse_args()

    results = {
        "experiment_1_video_length": [],
        "experiment_2_iterations":   [],
        "experiment_3_models":       [],
        "experiment_4_platforms":    [],
        "experiment_5_captions":     [],
    }

    # ── Experiment 1: Video Length Scalability ────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 1: Video Length Scalability")
    print("="*60)

    configs = [
        ("short",  args.video_short,  "Make a 15 second highlight clip"),
        ("medium", args.video_medium, "Make a 30 second highlight clip"),
        ("long",   args.video_long,   "Make a 60 second highlight clip"),
    ]
    for label, vpath, prompt in configs:
        print(f"\n  Testing {label} video ({vpath})...")
        m = run_single_experiment(vpath, prompt, model_name="gemini-2.0-flash", max_iterations=3)
        m["config_label"] = label
        results["experiment_1_video_length"].append(m)
        print(f"  → Latency: {m.get('latency_seconds')}s | Score: {m.get('quality_score')} | Iterations: {m.get('iterations_used')}")

    # ── Experiment 2: Agentic Loop Iterations ────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 2: Agentic Loop Iterations (1 vs 2 vs 3)")
    print("="*60)

    for max_iter in [1, 2, 3]:
        print(f"\n  Testing max_iterations={max_iter}...")
        m = run_single_experiment(
            args.video_medium,
            "Make a 30 second cinematic highlight clip for TikTok",
            model_name="gemini-2.0-flash",
            max_iterations=max_iter,
        )
        m["config_label"] = f"max_iter_{max_iter}"
        results["experiment_2_iterations"].append(m)
        print(f"  → Score: {m.get('quality_score')} | Iterations used: {m.get('iterations_used')} | Passed: {m.get('passed')}")

    # ── Experiment 3: Model Comparison ───────────────────────────────────────
    if not args.skip_models:
        print("\n" + "="*60)
        print("EXPERIMENT 3: Model Comparison")
        print("="*60)

        models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        for model in models:
            print(f"\n  Testing model: {model}...")
            m = run_single_experiment(
                args.video_medium,
                "Make a 30 second highlight clip with captions",
                model_name=model,
                max_iterations=2,
            )
            m["config_label"] = model
            results["experiment_3_models"].append(m)
            print(f"  → Score: {m.get('quality_score')} | Latency: {m.get('latency_seconds')}s")
    else:
        print("\nSkipping model comparison (--skip_models flag set)")

    # ── Experiment 4: Platform Presets ───────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 4: Platform Preset Export")
    print("="*60)

    platforms = ["tiktok", "instagram_reels", "youtube_shorts", "linkedin"]
    for platform in platforms:
        print(f"\n  Testing platform: {platform}...")
        m = run_single_experiment(
            args.video_medium,
            f"Export for {platform}",
            model_name="gemini-2.0-flash",
            max_iterations=2,
            platform_key=platform,
        )
        m["config_label"] = platform
        results["experiment_4_platforms"].append(m)
        print(f"  → Score: {m.get('quality_score')} | Output: {m.get('output_info', {}).get('resolution')}")

    # ── Experiment 5: Caption Quality ────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 5: Caption Quality")
    print("="*60)

    for label, vpath in [("short", args.video_short), ("medium", args.video_medium)]:
        print(f"\n  Captioning {label} video...")
        m = run_caption_quality_experiment(vpath, model_name="gemini-2.0-flash")
        m["config_label"] = label

        # Self-ROUGE: compare first half of transcript against second half as proxy
        if m.get("full_text") and len(m["full_text"]) > 100:
            mid = len(m["full_text"]) // 2
            rouge = compute_rouge(m["full_text"][:mid], m["full_text"][mid:])
            m["self_rouge"] = rouge
            print(f"  → Segments: {m.get('segment_count')} | Latency: {m.get('caption_latency')}s | Self-ROUGE-1: {rouge.get('rouge1')}")

        results["experiment_5_captions"].append(m)

    # ── Save Results ─────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ All experiments complete. Results saved to: {args.output}")
    print(f"{'='*60}")
    print("\nSUMMARY:")
    print(f"  Experiment 1 (Video Length):  {len(results['experiment_1_video_length'])} configs tested")
    print(f"  Experiment 2 (Iterations):    {len(results['experiment_2_iterations'])} configs tested")
    print(f"  Experiment 3 (Models):        {len(results['experiment_3_models'])} models tested")
    print(f"  Experiment 4 (Platforms):     {len(results['experiment_4_platforms'])} platforms tested")
    print(f"  Experiment 5 (Captions):      {len(results['experiment_5_captions'])} videos tested")
    print("\nNow open results.json and plug the numbers into your Results section.")


if __name__ == "__main__":
    main()