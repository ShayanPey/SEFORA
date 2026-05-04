from openai import OpenAI
import time
import os

client = OpenAI(api_key=os.environ["OPENAI_KEY"])


GPT_SEGMENT_TEMPLATE = (
    """An instructor left feedback for a specific paragraph of a student's essay. 
    Your task is to segment this feedback into single feedback points (also called feedback units).

    A single feedback point (also referred to as a feedback unit) is defined as a distinct, self-contained statement that addresses one specific aspect of a student's writing. We treat a feedback point as one or more sentences that convey a coherent thought and can be understood independently . Importantly, a single feedback message may contain multiple such points, each corresponding to a different issue or suggestion.

    Characteristics of a single feedback point:
    - Specificity: Focuses on one aspect of writing (e.g., grammar, word choice, structure, content, tone, a character in the story, etc.).
    - Coherence: Forms a complete thought that is interpretable without requiring additional context.
    - Actionability: It may offer an identifiable issue or suggestion that the student could address.

    If a span elaborates on the same aspect of writing (e.g., further explaining or justifying the same critique or suggestion), it remains part of the same feedback point. If the span shifts to a different aspect, comment, or suggestion, it should be marked as a new feedback point with a delimiter.
    
    Task:
        - Insert $$ (double dollar-sign) as a delimiter when the feedback moves to a new aspect of the student's writing.
        - Each segment should be able to stand alone as a complete feedback point.
        - Do not insert delimiters for mere elaboration, clarification, or examples of the same point.
        - Do not use any other characters (spaces, line breaks, tabs). Only insert $$.
        - If there is a need to place a delimiter in the middle of a sentence, and if the feedback points are
        separated by a transition word (e.g. "and", "but", "also", etc.), then place the delimiter before the
        transition word.
        - Every delimiter used must be either at the end of the feedback, or adjacent to a space (or tab or a
        linebreak character). The only case where a delimiter may be placed between two non-space
        characters is when in the original sample, there is no space after the punctuation of the previous
        feedback point.
    
    Samples:
        Input 1:
            <input>Nice! This is well-written. But you should make it more concise, and be consistent with your characters, like who is Alex here?</input>
        Output 1:
            Nice! This is well-written.$$ But you should make it more concise,$$ and be consistent with your characters, like who is Alex here?$$
        Note 1: There are three feedback units here, one saying this is well-written, another saying make it more concise, and other is talking about characters. So we place a delimiter to segment them.

        Input 2:
            <input>However, the sentence structure is a bit choppy. For example, "As I started to approach, I made eye contact with Tyler." Could use smoother transitions between phrases.</input>
        Output 2:
            However, the sentence structure is a bit choppy. For example, "As I started to approach, I made eye contact with Tyler." Could use smoother transitions between phrases.$$
        Note 2: This is just one feedback unit. Second sentence is just giving an example for the same feedback, and the last sentence is just wrapping up the point. So we just place one delimiter at the end, and nothing in between.
        
        Input 3:
            <input>What did he exactly see? Why did he decide to leave? What about Sara... did she leave too? You need to revise this paragraph.</input>
        Output 3:
            What did he exactly see? Why did he decide to leave? What about Sara... did she leave too? You need to revise this paragraph.$$
        Note 3: This is one feedback point. Although it asks many questions, but basically the feedback is saying that more elaboration and clarification is needed. The writer needs to add details to make it more clear. So we DON'T insert a delimiter for every question. Of course, the questions are asking different things, but this is one feedback unit. They are ALL talking about the same aspect of the essay.
        
        Input 4:
            <input>The sensory details are just great, and the conversation between the characters, really engaging!</input>
        Output 4:
            The sensory details are just great,$$ and the conversation between the characters, really engaging!$$
        Note 4: These are two feedback units, since first is talking about the sensory details and saying it's great, and the second feedback unit is talking about the conversation between the characters being engaging.
        
        Input 5:
            <input>I don't quite understand what did he tell him? What did Sara think about this? Also this paragraph is a little verbose.</input>
        Output 5:
            I don't quite understand what did he tell him? What did Sara think about this?$$ Also this paragraph is a little verbose.$$
        Note 5: We can see that there are only 2 feedback units. First is about something being unclear, and second feedback is saying it's verbose.
        
        Input 6:
            <input>You could add more sensory details. Like explain what was different about the sky that night? These details help your paragraph be more effective.</input>
        Output 6:
            You could add more sensory details. Like explain what was different about the sky that night? These details help your paragraph be more effective.$$
        Note 6: If we place delimiter anywhere except at the end, we would've broken the most important rule, which is the feedback units must be irrelevant. They are related. They are all talking about sensory details. The second sentence is just specifying a specific part, but asking for sensory details (again). The last sentence is just pointing out why sensory details are important. SO WE DON'T PLACE A DELIMITER TO DISTINGUISH THESE SENTENCES AS ALL BELONG TO THE SAME FEEDBACK POINT.
        
        Using the guidelines above, segment the following feedback into feedback units. Don't provide any explanation; just the same input with delimiters placed in correct place(s). 
    Input: 
    <input>{llm_feedback}</input>
    Segmented Feedback:
    """
)

def llm_segment_feedback(feedback: str, printApiLog: bool=False) -> str:
    """
    Calls GPT to segment a feedback string into feedback points using GPT_SEGMENT_TEMPLATE.
    Retries up to 5 times on failure.
    """
    if printApiLog:
        print("\n\nGPT API SEGMENT:\n", feedback[:30], "...", end=" ")

    for attempt in range(5):
        try:
            response = client.responses.create(
            model="gpt-5-nano",
            reasoning={"effort": "low"},
            input=[
                {
                    "role": "user", 
                    "content": GPT_SEGMENT_TEMPLATE.format(llm_feedback=feedback)
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


def segment_files(
    source_dir: str,
    dest_dir: str,
    num_files: Optional[int] = None,
    continue_ind: Optional[int] = None,
    printApiLog: bool = False,
    sub56: bool = False,
):
    """
    Copies JSON files from source_dir -> dest_dir.

    For each paragraph where:
      - para["model"] exists and is non-empty
      - para["model"] has exactly ONE item
      - model[0]["feedback"] contains a string starting with "POSTPROCESS: "

    It:
      - extracts the POSTPROCESS content (prefix removed)
      - sends it to llm_segment_feedback(str) -> str
      - appends "FEEDBACK_SEGMENTS" to feedback
      - splits the returned string on "$$"
      - appends each segment that:
            * stripped length >= 3

    All other content is copied identically.
    """

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"source_dir not found or not a directory: {source_dir}")

    os.makedirs(dest_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(source_dir) if f.lower().endswith(".json"))

    start = 0 if continue_ind is None else int(continue_ind)
    if start < 0 or start > len(files):
        raise ValueError(f"continue_ind out of range: {start} (num files: {len(files)})")

    if num_files is None:
        selected = files[start:]
    else:
        n = int(num_files)
        if n < 0:
            raise ValueError(f"num_files must be >= 0 or None, got: {n}")
        selected = files[start:n]

    processed_count = 0
    
    sub56_path = "sub56.txt"
    if sub56:
        if not os.path.isfile(sub56_path):
            raise FileNotFoundError(f"{sub56_path} not found")
        with open(sub56_path, "r", encoding="utf-8") as f:
            sub56_names = {line.strip() for line in f if line.strip()}

    for fname in selected:
        print(f"PROCESSING: {processed_count + 1} / {len(selected)} {fname}")
        
        if sub56 and not (fname in sub56_names):
            print("NOT SELECTED... SKIPPED...")
            processed_count +=1
            continue
        print("SELECTED... Continue processing...")

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
            paragraphs = []

        changed = False

        for p_idx, para in enumerate(paragraphs):
            if not isinstance(para, dict):
                continue

            model_list = para.get("model")
            if not isinstance(model_list, list) or len(model_list) != 1:
                continue

            model0 = model_list[0]
            if not isinstance(model0, dict):
                continue

            feedback = model0.get("feedback")
            if not isinstance(feedback, list):
                continue

            # Find POSTPROCESS entry
            postprocess_str = None
            for fb in feedback:
                if isinstance(fb, str) and fb.startswith("POSTPROCESS: "):
                    postprocess_str = fb[len("POSTPROCESS: "):]
                    break

            if postprocess_str is None:
                continue

            # Call segmentation function
            try:
                segmented = llm_segment_feedback(postprocess_str, printApiLog)
            except Exception as e:
                print(f"[ERROR] {fname}: llm_segment_feedback failed at paragraph {p_idx}: {e}")
                continue

            if not isinstance(segmented, str):
                continue

            feedback.append("FEEDBACK_SEGMENTS")

            parts = segmented.split("$$")
            for part in parts:
                s = part.strip()
                if len(s) >= 3:
                    feedback.append(s)

            changed = True

        try:
            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to write {fname}: {e}")
            continue

        processed_count += 1
        print(f"[OK] {fname} -> written ({'changed' if changed else 'no eligible paragraphs'})")

    print(f"\nDone. Wrote {processed_count} file(s) to: {dest_dir}")


import sys

if len(sys.argv) != 2:
    raise RuntimeError("Usage: python p.py <setting>")

setting = sys.argv[1]

setting = 's' + setting

segment_files("./postprocess_individual/pp_"+setting,
                   "./segment_individual/seg_"+setting,
                   printApiLog = False,
                   num_files=None,
                   continue_ind=None,
                   sub56 = False)