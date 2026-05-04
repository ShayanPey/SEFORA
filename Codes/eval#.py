import time
import os

from google import genai
from google.genai import types


gem_client = genai.Client(api_key=os.environ["GEMINI_KEY"])



GPT_SIM_BATCH_PROMPT = """Problem Statement
Given two feedback units targeted to a paragraph of an essay, how “similar” are they? A feedback unit is defined as a self-contained statement that addresses one specific aspect of a student’s writing. 

Guideline
Given a pair of feedback units, A and B, annotate their “similarity” on a scale of 0 to 4.

Note 0: We assume that both feedback units are given on the same essay and the same paragraph. If one unit provides a comment without explicitly specifying which part of the story or essay it targets, while the other does, it is safe to assume they refer to the same part, as paragraphs typically cover only a specific portion of the story, essay, or a specific scene.

Note 1: This is not literal semantic equivalence. Similarity is driven primarily by whether the units target the same aspect/issue and/or they convey the same comment of the writing. Targeting the same aspect establishes a high baseline (~2), and alignment in what the feedback recommends/implies pushes the score up; opposition/contradiction pushes it down toward 0. And vice versa, i.e. they have the same comment, establish a high baseline (~2/3), and alignment in the targeted aspect increases the score towards 4.

Score 4 is the highest score, indicating that A and B are almost equivalent as feedback. This represents a looser standard than literal semantic equivalence, as the feedback units may be phrased differently, but the underlying “point” they convey is the same.

Score 0 is the lowest score, indicating that A and B are completely irrelevant to each other or entirely contradictory as feedback.

Conceptually, scores 1, 2, and 3 are defined as equally spaced points between the two ends of the spectrum: score 0 (completely irrelevant) and score 4 (almost equivalent).

Score 3 represents that A and B are very close as feedback, but a detail differs. For instance, if they convey the same point (a similar comment), but one of them targets a more specific aspect that is a part of the general aspect the other unit is focusing on. 

Score 2 indicates that A and B share some important/main points, but also differ on other important/main points. For example, if both focus on the same aspect of the writing but provide different comments, they share an important point (same important aspect) yet differ on the comments (different important points). Similarly, if A and B express the same comment or point but focus on different aspects of the writing, they again share one important point while differing on another.

Score 1 indicates that A and B are mostly irrelevant as feedback, but share a minor common detail and are not completely unrelated. For instance, both may mention an issue (shared detail), yet the issues themselves are entirely different (differing important points), resulting in a similarity score of 1.

Additional Notes: So we do know that these feedback are given towards the same paragraph of the same essay, we are measuring the similarity of these two feedback units.

Samples of each score:
Score 4: (Almost) semantically equivalent as feedback units.
A: <feedback>Great job!</feedback>
B: <feedback>Wow, this paragraph is really well-written!</feedback>
-> They both convey generally/all aspects are good. As feedback, neither convey anything else.

A: <feedback>Too many repeated words.</feedback>
B: <feedback>Try to broaden the vocabulary used.</feedback>
-> They are just rephrased, and convey the same exact point.

A: <feedback>This opening is captivating!</feedback>
B: <feedback>Nice job setting the scene and creating suspense.</feedback>
-> Opening ≈ Setting the Scene (Same aspect). Captivating ≈ Nice job creating suspense (Same point/comment)

Score 3: Share the same important/main point(s), some details differ.
A: <feedback>Add more details about the scene.</feedback>
B: <feedback>What did he see exactly?</feedback>
-> Both feedback are asking for more details (shared important point), A is asking for general details, B is specifically asking for a specific missing detail (differing detail).

A: <feedback>The story is a little choppy</feedback>
B: <feedback>I couldn’t follow; why didn’t she leave then?</feedback>
-> Both convey the point of the story not having a proper flow and being abrupt (shared important point), B is addressing a specific part of the story that is abrupt.

A: <feedback>This is well-written. Nice job.</feedback>
B: <feedback>The sensory details are just perfect.</feedback>
-> They both convey the point of being well-written (shared important point) but one specifies exactly what is making it well-written (different detail).

A: <feedback>This is good…</feedback>
B: <feedback>Excellent!</feedback>
-> Both feedback are conveying the point of being well-written (shared important point), but their intensity is different (different detail)

Score 2: Share some important point(s), some important points differ.
A: <feedback>Such a balanced and well-written paragraph!</feedback>
B: <feedback>You well-developed the characters.</feedback>
-> They both convey the point of being well-written (shared important point), but B specifically mentions a main aspect of the essay, the characters (different important point).
Why not 3: Characters of a story cannot be considered as a detail. Characters are part of the paragraph, but the difference is not negligible. Also, the comment of “balanced and well-written” and “well-developed” are also different.

A: <feedback>Clarify what is exactly happening in this scene!</feedback>
B: <feedback>Please clarify what you mean by “blue smell”.</feedback>
-> Both are asking for clarification (shared important point), but B is asking for clarification of a specific phrase (different important point).
Why not 3: They are asking for clarification, which is very general (but is enough to set the baseline on score 2), but A asks for clarification of what is happening, B is asking for clarification of a phrase or a sensory detail (which is completely different and irrelevant).

A: <feedback>Very nice voice and tone.</feedback>
B: <feedback>Very well developed characters!</feedback>
-> They are both saying that an element of the essay is good (important shared point), but both specify a specific aspect of the essay (important different point). That is, the aspect of voice/tone [of the story] is almost completely irrelevant to the establishment of the characters.

Score 1: Share a few details, but important points differ. (Not COMPLETELY irrelevant, or they share the same general feedback category)
General category: Mistake, Elaboration/Clarification, Praise
A: <feedback>It should be “wear”, not “where”</feedback>
B: <feedback>Too much repetition...</feedback>
-> They share the same feedback category -- “mistake”

A: <feedback>Interesting conversation between you and Sara!</feedback>
B: <feedback>Nice job with the sensory details.</feedback>
-> Same feedback category -- “praise”, but the focus and the main elements of the feedback is irrelevant: sensory details of the scene, and the conversation between characters. Feedback A also only conveys the point that the conversation itself is interesting, and is not specifically saying this is well-written.

Score 0: Completely irrelevant or even opposite.
A: <feedback>This paragraph is great!</feedback>
B: <feedback>This paragraph is a little choppy</feedback>
-> They somewhat contradict each other. They also don’t share the same feedback category.

A: <feedback>Very well-written.</feedback>
B: <feedback>So why did she remain on the couch?</feedback>
-> Irrelevant: A is praising, B is asking for clarification about some part of the story.

A: <feedback>So much authenticity in your voice!</feedback>
B: <feedback>You can reduce the exaggeration here</feedback>
-> Irrelevant: A is praising the paragraph, while B is criticizing an aspect of the paragraph.

Now score EACH pair completely INDEPENDENTLY. Do NOT let the score of one pair influence another.

You will receive multiple labeled pairs. For EACH label, output exactly one line:

<label>) <integer 0-4>

No explanation. No extra text. Only those lines, one per label, in the same order.

PAIRS:
{pairs_block}
"""
GEMINI_SIM_BATCH_PROMPT=GPT_SIM_BATCH_PROMPT


import re
import time
from collections import deque
from typing import Deque, List, Optional, Tuple


def _make_tags(n: int) -> List[str]:
    # A, B, ..., Z, AA, AB, ...
    tags: List[str] = []
    i = 0
    while len(tags) < n:
        x = i
        s = ""
        while True:
            s = chr(ord("A") + (x % 26)) + s
            x = x // 26 - 1
            if x < 0:
                break
        tags.append(s)
        i += 1
    return tags


def _build_pairs_block(pairs: List[Tuple[str, str]]) -> str:
    tags = _make_tags(len(pairs))
    chunks: List[str] = []
    for tag, (f1, f2) in zip(tags, pairs):
        chunks.append(
            f"{tag})\n"
            f"Instructor A feedback: <feedback>{f1}</feedback>\n"
            f"Instructor B feedback: <feedback>{f2}</feedback>\n"
        )
    return "\n".join(chunks)


def _parse_tagged_scores(text: str, expected_tags: List[str]) -> Optional[List[int]]:
    # STRICT: reject duplicates, missing tags, extra tags.
    scores_by_tag: dict[str, int] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        m = re.match(r"^([A-Z]{1,10})\)\s*([0-4])\s*$", line)
        if not m:
            continue

        tag, score_str = m.group(1), m.group(2)

        # reject duplicates
        if tag in scores_by_tag:
            return None

        scores_by_tag[tag] = int(score_str)

    if len(scores_by_tag) != len(expected_tags):
        print("scores by tag:", len(scores_by_tag), "expected_tags", len(expected_tags), '\n\nscores by tag:', scores_by_tag, '\n\nexpected:', expected_tags)
        return None

    if set(scores_by_tag.keys()) != set(expected_tags):
        print("Sets do not match! scores keys:\n", scores_by_tag.keys(), "\nEXPECTED:\n", set(expected_tags))
        return None

    return [scores_by_tag[tag] for tag in expected_tags]


class GeminiSimilarityBuffered:
    """
    Buffers (feedback1, feedback2) pairs; sends to Gemini when buffer hits buffer_size
    or when flush=True is passed.

    add_pair(...) returns a LIST of newly-produced scores (possibly empty).
    Scores are returned in the same order as inputs were provided.
    """

    def __init__(
        self,
        gem_client,
        buffer_size: int,
        #model: str = "gemini-3-flash-preview",
        model: str = 'gemini-3.1-flash-lite-preview',
        prompt_template: str = GEMINI_SIM_BATCH_PROMPT, 
        printApiLog: bool = False,
        max_attempts: int = 5,
        max_parse_retries: int = 5,
        sleep_between_attempts_s: float = 2.0,
        sleep_between_parse_retries_s: float = 2.0,
    ):
        if buffer_size <= 0:
            raise ValueError("buffer_size must be >= 1")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be >= 1")
        if max_parse_retries <= 0:
            raise ValueError("max_parse_retries must be >= 1")
        if not prompt_template:
            raise ValueError("prompt_template must be a non-empty string")

        self.gem_client = gem_client
        self.buffer_size = buffer_size
        self.model = model
        self.prompt_template = prompt_template
        self.printApiLog = printApiLog
        self.max_attempts = max_attempts
        self.max_parse_retries = max_parse_retries
        self.sleep_between_attempts_s = sleep_between_attempts_s
        self.sleep_between_parse_retries_s = sleep_between_parse_retries_s

        self._buffer: List[Tuple[str, str]] = []
        self._pending: Deque[int] = deque()

    def _extract_text(self, response) -> str:
        # Matches your single-pair usage: response.text
        t = getattr(response, "text", None)
        if isinstance(t, str):
            return t.strip()

        # Some SDK variants return a list of candidates/parts; safest fallback:
        try:
            return str(response).strip()
        except Exception:
            return ""

    def add_pair(
        self,
        feedback1: Optional[str],
        feedback2: Optional[str],
        flush: bool = False,
    ) -> List[int]:
        produced: List[int] = []

        # Drain any pending scores first
        while self._pending:
            produced.append(self._pending.popleft())

        # Add current pair unless this call is flush-only
        if feedback1 is not None and feedback2 is not None:
            self._buffer.append((feedback1, feedback2))

        should_send = flush or (len(self._buffer) >= self.buffer_size)
        if not should_send:
            if self.printApiLog and feedback1 is not None and feedback2 is not None:
                print(feedback1[:10], "----", feedback2[:10], " Buffered...")
            return produced  # likely empty

        # If flush=True and buffer is empty, nothing to do
        if not self._buffer:
            return produced

        batch_pairs = self._buffer
        self._buffer = []

        tags = _make_tags(len(batch_pairs))
        pairs_block = _build_pairs_block(batch_pairs)
        prompt_text = self.prompt_template.format(pairs_block=pairs_block)

        if self.printApiLog:
            print(f"GEMINI Model: {self.model} | batch={len(batch_pairs)}")

        parsed: Optional[List[int]] = None
        parse_tries = 0

        for attempt in range(self.max_attempts):
            try:
                response = self.gem_client.models.generate_content(
                    model=self.model,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level="minimal"
                        )
                    ),
                )

                out_text = self._extract_text(response)


                parsed = _parse_tagged_scores(out_text, tags)
                if parsed is None:
                    parse_tries += 1
                    if self.printApiLog:
                        preview = out_text.replace("\n", " ")[:]
                        print(
                            f"Bad parse (parse_try {parse_tries}/{self.max_parse_retries}) | {preview}"
                        )
                        print("\n\n\nSENT PROMPT:", prompt_text)

                    # Retry parse by re-calling the model
                    if parse_tries < self.max_parse_retries:
                        time.sleep(self.sleep_between_parse_retries_s)
                        continue

                    # Too many bad parses -> fallback zeros
                    parsed = [0] * len(batch_pairs)

                # Valid parsed (or fallback zeros)
                for s in parsed:
                    self._pending.append(s)
                break

            except Exception as e:
                if self.printApiLog:
                    print(f"Attempt {attempt + 1} failed:", e)
                time.sleep(attempt*self.sleep_between_attempts_s)

        if parsed is None:
            # All API attempts failed
            for _ in batch_pairs:
                self._pending.append(0)

        while self._pending:
            produced.append(self._pending.popleft())

        return produced




import os
import json
from collections import deque
from typing import Optional, Dict, List, Tuple, Deque, Any

def stage1_score_and_save_pairs(
    *,
    model_source: str,
    annot_source: str,
    scores_jsonl_path: str,
    scorer, 
    num_files: Optional[int] = None,
    continue_ind: Optional[int] = None,
    printApiLog: bool = False,
    sub56: bool = False,
    one_GT_only: bool = False,
    g1_GT_only: bool = False,
) -> Dict[str, int]:
    """
    Iterates like segmatch, but instead of instant API calls it uses the provided
    buffered scorer (GPTSimilarityBuffered) and writes ONE JSONL record per pair.

    JSONL fields per line:
      file, paragraph_index, model_seg_index, annot_index, feedback, comment, score

    Returns counts for sanity checking.
    """
    if not os.path.isdir(model_source):
        raise FileNotFoundError(f"model_source not found: {model_source}")
    if not os.path.isdir(annot_source):
        raise FileNotFoundError(f"annot_source not found: {annot_source}")
    
    if g1_GT_only and one_GT_only:
        raise ValueError("Can't set both one_GT_only and g1_GT_only to True.")
    
    if g1_GT_only:
        scores_jsonl_path = scores_jsonl_path[:-6]+"_g1GT.jsonl"

    out_dir = os.path.dirname(os.path.abspath(scores_jsonl_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(model_source) if f.lower().endswith(".json"))
    start = 0 if continue_ind is None else int(continue_ind)
    selected = files[start:] if num_files is None else files[start:start + int(num_files)]

    sub56_names: Optional[set[str]] = None
    if sub56:
        with open("sub56.txt", "r", encoding="utf-8") as f:
            sub56_names = {line.strip() for line in f if line.strip()}

    pending_meta: Deque[Dict[str, Any]] = deque()

    total_files_seen = 0
    total_files_scored = 0
    total_pairs_added = 0
    total_pairs_written = 0

    def _write_scored_items(fh, scores: List[int]) -> None:
        nonlocal total_pairs_written
        for s in scores:
            if not pending_meta:
                raise RuntimeError(
                    "Internal mismatch: got a score from scorer but no pending metadata. "
                    "This indicates a logic error in buffering alignment."
                )
            meta = pending_meta.popleft()
            meta["score"] = int(s)
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
            total_pairs_written += 1

    with open(scores_jsonl_path, "w", encoding="utf-8") as out_f:
        for i, fname in enumerate(selected, 1):
            total_files_seen += 1
            if printApiLog:
                print(f"STAGE1 PROCESSING: {i}/{len(selected)} {fname}")

            if sub56 and sub56_names is not None and fname not in sub56_names:
                if printApiLog:
                    print("STAGE1 SKIPPED (sub56)...")
                continue

            model_path = os.path.join(model_source, fname)
            annot_path = os.path.join(annot_source, fname)

            if not os.path.isfile(annot_path):
                if printApiLog:
                    print(f"[STAGE1 ERROR] Missing annotation file: {fname}")
                continue

            with open(model_path, "r", encoding="utf-8") as f:
                model_data = json.load(f)
            with open(annot_path, "r", encoding="utf-8") as f:
                annot_data = json.load(f)

            wrote_any_for_file = False

            for p_idx, (p_model, p_annot) in enumerate(
                zip(model_data.get("paragraphs", []), annot_data.get("paragraphs", []))
            ):
                model_list = p_model.get("model")
                if not isinstance(model_list, list) or len(model_list) != 1:
                    continue

                feedback = model_list[0].get("feedback")
                if not isinstance(feedback, list) or "FEEDBACK_SEGMENTS" not in feedback:
                    continue

                seg_idx = feedback.index("FEEDBACK_SEGMENTS")
                model_segs = [
                    s for s in feedback[seg_idx + 1:]
                    if isinstance(s, str) and len(s.strip()) >= 3
                ]
                if not model_segs:
                    continue

                comments = [
                    a.get("comment", "").strip()
                    for a in p_annot.get("annotations", [])
                    if isinstance(a, dict) and len(a.get("comment", "").strip()) >= 3
                ]
                if not comments:
                    continue

                if one_GT_only and len(comments) != 1:
                    continue
                
                if g1_GT_only and  len(comments) == 1:
                    continue

                for mi, ms in enumerate(model_segs):
                    for ai, ac in enumerate(comments):
                        pending_meta.append({
                            "file": fname,
                            "paragraph_index": p_idx,
                            "model_seg_index": mi,
                            "annot_index": ai,
                            "feedback": ms,
                            "comment": ac,
                        })
                        total_pairs_added += 1
                        wrote_any_for_file = True

                        produced = scorer.add_pair(ms, ac, flush=False)
                        if produced:
                            _write_scored_items(out_f, produced)

            if wrote_any_for_file:
                total_files_scored += 1

        # flush leftovers
        produced = scorer.add_pair(None, None, flush=True)
        if produced:
            _write_scored_items(out_f, produced)

    if pending_meta:
        raise RuntimeError(
            f"Stage1 ended but still has {len(pending_meta)} pending metadata items. "
            "This indicates a mismatch between pairs added and scores produced."
        )

    return {
        "files_seen": total_files_seen,
        "files_scored": total_files_scored,
        "pairs_added": total_pairs_added,
        "pairs_written": total_pairs_written,
    }


    
    
import sys

if len(sys.argv) != 2:
    raise RuntimeError("Usage: python file.py <setting>")

setting = sys.argv[1]

setting = 's' + setting


scorer = GeminiSimilarityBuffered(
    gem_client=gem_client,
    buffer_size=50,
    model="gemini-3.1-flash-lite-preview",
    prompt_template=GEMINI_SIM_BATCH_PROMPT,
    printApiLog=True,
)


stats = stage1_score_and_save_pairs(
    model_source="./segment_individual/seg_"+setting,
    annot_source="./segment_individual/seg_annot/",
    scores_jsonl_path="./segmatch_individual/gemini_buffer_stage1_scores_"+setting+".jsonl",
    scorer=scorer,
    num_files=None,
    continue_ind=None,
    printApiLog=True,
    sub56=False,
    one_GT_only=True,
    g1_GT_only=False,
)

print("STAGE1 STATS:", stats)