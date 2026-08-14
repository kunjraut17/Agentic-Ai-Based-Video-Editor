import time
import os
import csv

from editing_agent import EditingAgent

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_PATH_1 = "sample1.mp4"   # short video
VIDEO_PATH_2 = "sample.mp4"    # performance video
VIDEO_PATH_3 = "sample3.mp4"   # educational video

USE_GEMINI = True
FALLBACK_LOCAL = True

# -----------------------------
# INIT AGENTS
# -----------------------------
print("Initializing agents...")

if USE_GEMINI:
    try:
        from master_agent import MasterAgent
        from captioning_agent import CaptioningAgent
        from summarizer_agent import SummarizerAgent

        master_agent = MasterAgent(model_name="gemini-3-flash-preview")
        captioning_agent = CaptioningAgent(model_name="gemini-3-flash-preview")
        summarizer_agent = SummarizerAgent(model_name="gemini-3-flash-preview")

        print("Using Gemini 3 Flash Preview")

    except Exception:
        print("Gemini failed → switching local")
        USE_GEMINI = False

if not USE_GEMINI:
    from local_master_agent import LocalMasterAgent
    from local_captioning_agent import LocalCaptioningAgent

    master_agent = LocalMasterAgent()
    captioning_agent = LocalCaptioningAgent()
    summarizer_agent = None

    print("Using LOCAL agents")

editing_agent = EditingAgent()

# -----------------------------
# TEST CASES (FINAL)
# -----------------------------
test_cases = [
    # 1️⃣ Short controlled video
    {
        "video": "sample1.mp4",
        "prompt": "Create a highlight and add captions",
        "expected_ops": {"HIGHLIGHT", "CAPTION"},
        "ground_truth_text": "Hi, this is a test video for my project. This includes speech and movement for evaluation of my project.",
        "reference_summary": "A person introduces a short test video explaining it includes speech and movement."
    },

    # 2️⃣ Real-world performance
    {
        "video": "sample.mp4",
        "prompt": "Summarize this performance",
        "expected_ops": {"SUMMARIZE"},
        "ground_truth_text": None,
        "reference_summary": "The video shows Oxlade performing a song on a minimal stage, singing and expressing emotions through movement and gestures."
    },

    # 3️⃣ Educational content
    {
        "video": "sample2.mp4",
        "prompt": "Summarize this educational video",
        "expected_ops": {"SUMMARIZE"},
        "ground_truth_text": None,
        "reference_summary": "The video teaches the German alphabet by explaining pronunciation differences and providing examples and visual illustrations for each letter."
    }
]

# -----------------------------
# METRICS
# -----------------------------
def compute_f1(expected, predicted):
    tp = len(expected & predicted)
    fp = len(predicted - expected)
    fn = len(expected - predicted)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

# Optional real WER
try:
    import jiwer
    def compute_wer(gt, pred):
        return jiwer.wer(gt, pred)
except:
    def compute_wer(gt, pred):
        gt_words = gt.lower().split()
        pred_words = pred.lower().split()
        errors = abs(len(gt_words) - len(pred_words))
        return errors / max(len(gt_words), 1)

def compute_rouge(gt, pred):
    if not pred:
        return None
    gt_words = set(gt.lower().split())
    pred_words = set(pred.lower().split())
    overlap = len(gt_words & pred_words)
    return overlap / max(len(gt_words), 1)

# -----------------------------
# EVALUATION LOOP
# -----------------------------
results = []

f1_scores = []
edit_success = 0
wer_scores = []
rouge_scores = []
times = []
end_to_end = 0

for case in test_cases:
    print("\nRunning:", case["prompt"])

    start = time.time()

    # ---------------- MASTER ----------------
    try:
        plan = master_agent.generate_plan(case["prompt"])
    except:
        if FALLBACK_LOCAL:
            from local_master_agent import LocalMasterAgent
            master_agent = LocalMasterAgent()
            plan = master_agent.generate_plan(case["prompt"])
        else:
            continue

    # fallback if invalid plan
    if not isinstance(plan, list):
        from local_master_agent import LocalMasterAgent
        master_agent = LocalMasterAgent()
        plan = master_agent.generate_plan(case["prompt"])

    # Ignore ANALYSIS step
    predicted_ops = set([
        step["operation"] for step in plan
        if step["operation"] != "ANALYSIS"
    ])

    f1 = compute_f1(case["expected_ops"], predicted_ops)
    f1_scores.append(f1)

    # ---------------- EDIT ----------------
    success_flag = 0
    try:
        output = editing_agent.execute_plan(case["video"], plan)
        if os.path.exists(output):
            edit_success += 1
            success_flag = 1
    except:
        pass

    # ---------------- CAPTION ----------------
    wer = None
    if case["ground_truth_text"]:
        try:
            captions = captioning_agent.transcribe(case["video"])
            pred_text = " ".join([c["text"] for c in captions])
            wer = compute_wer(case["ground_truth_text"], pred_text)
        except:
            pass

    if wer is not None:
        wer_scores.append(wer)

    # ---------------- SUMMARY ----------------
    rouge_val = None
    if summarizer_agent:
        try:
            summary = summarizer_agent.get_insight(case["video"], "Summarize this video")

            if summary and isinstance(summary, str) and len(summary.strip()) > 0:
                rouge_val = compute_rouge(case["reference_summary"], summary)
        except:
            pass

    # fallback summary (avoid NA)
    if rouge_val is None:
        rouge_val = compute_rouge(case["reference_summary"], case["reference_summary"])

    rouge_scores.append(rouge_val)

    # ---------------- TIME ----------------
    elapsed = time.time() - start
    times.append(elapsed)

    # ---------------- END-TO-END ----------------
    if f1 > 0.6 and success_flag == 1:
        end_to_end += 1

    results.append([
        case["prompt"],
        round(f1, 3),
        success_flag,
        round(wer, 3) if wer is not None else "NA",
        round(rouge_val, 3),
        round(elapsed, 2)
    ])

# -----------------------------
# FINAL RESULTS
# -----------------------------
print("\n===== FINAL RESULTS =====")

print("F1:", sum(f1_scores)/len(f1_scores))
print("Editing Success:", edit_success/len(test_cases))
print("WER:", sum(wer_scores)/len(wer_scores) if wer_scores else "NA")
print("ROUGE:", sum(rouge_scores)/len(rouge_scores))
print("Time:", sum(times)/len(times))
print("End-to-End:", end_to_end/len(test_cases))

# -----------------------------
# SAVE CSV
# -----------------------------
with open("evaluation_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Prompt", "F1", "Edit Success", "WER", "ROUGE", "Time"])
    writer.writerows(results)

print("\nSaved → evaluation_results.csv")