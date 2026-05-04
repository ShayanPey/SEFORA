from openai import OpenAI
import time
import os

client = OpenAI(api_key=os.environ["OPENAI_KEY"])

GPT_POST_PROCESS_TEMPLATE = (
"""You are given feedback written by an LLM about a specific paragraph of a student's essay. The feedback may include irrelevant text such as instructions, meta-commentary, formatting artifacts, or section titles.

Your task is to extract and return only the actual feedback directed at the student's paragraph.

Guidelines:
- Keep all relevant feedback exactly as written. Do not paraphrase, modify, or correct it.
- Remove any unrelated content, including:
  - Instructions (e.g., "step by step", "the user asked me to...")
  - Meta-commentary about the task or process
  - Section headers or titles
  - Formatting symbols (e.g., '*', '#')
- Be careful: some feedback may be very short (e.g., a single word like "were") but it probably refers to the student's writing. You are to remove the text that you understand to be irrelevant.
- Preserve original line breaks between relevant sentences.
- If the input is already clean, return it unchanged.
- If no relevant feedback is present, return an empty string.

Input:
{llm_feedback}

Output:
Return only the cleaned feedback, with no additional text.
"""
)

def llm_post_process(feedback: str, printApiLog: bool=False) -> str:
    """
    Calls GPT to post-process a feedback string using GPT_POST_PROCESS_TEMPLATE.
    Retries up to 5 times on failure.
    """
    if printApiLog:
        print("GPT API POST_PROCESS:", feedback[:30], "...", end=" ")

    for attempt in range(5):
        try:
            response = client.responses.create(
            model="gpt-5-mini",
            reasoning={"effort": "low"},
            input=[
                {
                    "role": "user", 
                    "content": GPT_POST_PROCESS_TEMPLATE.format(llm_feedback=feedback)
                }
            ]
            )
            out_tmp = response.output_text.strip()

            if printApiLog:
                print("✓")
            return out_tmp

        except Exception as e:
            if printApiLog:
                print(f"\nAttempt {attempt + 1} failed: {e}")
            time.sleep(2)

    if printApiLog:
        print("All GPT attempts failed. Returning original feedback...")
    return feedback  # fallback: return original text unchanged


import os
import json
from typing import Optional

def post_process_files(
    source_dir: str,
    dest_dir: str,
    num_files: Optional[int] = None,
    continue_ind: Optional[int] = None,
    printApiLog: bool = False,
    sub56: bool = False
):
    """
    Copies JSON files from source_dir -> dest_dir.
    For each paragraph where `para["model"]` exists and is non-empty (expected: exactly 1 item),
    takes `para["model"][0]["feedback"][0]`, runs `llm_post_process(...)`,
    and appends `POSTPROCESS: <result>` to `para["model"][0]["feedback"]`.

    Args:
        source_dir: directory containing .json files
        dest_dir: directory to write processed copies
        num_files: how many files to process (None => all remaining)
        continue_ind: start index in the sorted file list (None => 0)
    """
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"source_dir not found or not a directory: {source_dir}")

    os.makedirs(dest_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(".json")])

    start = 0 if continue_ind is None else int(continue_ind)
    if start < 0 or start > len(files):
        raise ValueError(f"continue_ind out of range: {start} (num files: {len(files)})")

    if num_files is None:
        selected = files[start:]
    else:
        n = int(num_files)
        if n < 0:
            raise ValueError(f"num_files must be >= 0 or None, got: {n}")
        selected = files[start : n]

    processed_count = 0
    sub56_path = "sub56.txt"
    if sub56:
        if not os.path.isfile(sub56_path):
            raise FileNotFoundError(f"{sub56_path} not found")
        with open(sub56_path, "r", encoding="utf-8") as f:
            sub56_names = {line.strip() for line in f if line.strip()}

    for fname in selected:
        print("PROCESSING:", processed_count+1, '/', len(selected), fname)
        if sub56 and not (fname.split('/')[-1] in sub56_names):
            print("NOT SELECTED... SKIPPED...")
            continue
        src_path = os.path.join(source_dir, fname)
        dst_path = os.path.join(dest_dir, fname)

        try:
            with open(src_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read {fname}: {e}")
            continue

        paragraphs = data.get("paragraphs", [])
        if not isinstance(paragraphs, list):
            print(f"[WARN] {fname}: 'paragraphs' is not a list; copying without changes.")
            paragraphs = []

        changed = False

        for p_idx, para in enumerate(paragraphs):
            if not isinstance(para, dict):
                print(f"[WARN] {fname}: paragraphs[{p_idx}] not a dict; skipping.")
                continue

            model_list = para.get("model", None)
            if not model_list:
                # model_list is missing, None, or empty list -> do nothing
                continue

            if not isinstance(model_list, list):
                print(f"[WARN] {fname}: paragraphs[{p_idx}].model is not a list; skipping.")
                continue

            if len(model_list) != 1:
                print(f"[WARN] {fname}: paragraphs[{p_idx}].model length != 1; skipping.")
                continue

            model0 = model_list[0]
            if not isinstance(model0, dict):
                print(f"[WARN] {fname}: paragraphs[{p_idx}].model[0] not a dict; skipping.")
                continue

            feedback = model0.get("feedback", None)
            if not isinstance(feedback, list) or len(feedback) == 0:
                print(f"[WARN] {fname}: paragraphs[{p_idx}].model[0].feedback missing/empty; skipping.")
                continue

            first_fb = feedback[0]
            if not isinstance(first_fb, str):
                print(f"[WARN] {fname}: paragraphs[{p_idx}].model[0].feedback[0] not a string; skipping.")
                continue

            # Call user-provided function
            try:
                out = llm_post_process(first_fb, printApiLog=printApiLog)
            except Exception as e:
                print(f"[ERROR] {fname}: llm_post_process failed at paragraphs[{p_idx}]: {e}")
                continue

            if not isinstance(out, str):
                print(f"[WARN] {fname}: llm_post_process returned non-string at paragraphs[{p_idx}]; skipping append.")
                continue

            feedback.append("POSTPROCESS: " + out)
            model0["feedback"] = feedback
            changed = True

        try:
            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to write {fname} to dest_dir: {e}")
            continue

        processed_count += 1
        print(f"[OK] {fname} -> written ({'changed' if changed else 'no paragraph model to process'})")

    print(f"\nDone. Wrote {processed_count} file(s) to: {dest_dir}")


import sys

if len(sys.argv) != 2:
    raise RuntimeError("Usage: python p.py <setting>")

setting = sys.argv[1]

setting = 's' + setting

post_process_files("./settings_individual/"+setting,
                   "./postprocess_individual/pp_"+setting,
                   printApiLog = False,
                   num_files=None,
                   continue_ind=None,
                   sub56=False)