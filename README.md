# SEFORA
This repository accompanies our paper, **SEFORA**: **S**tudent **E**ssays with **F**eedback C**or**pous **A**nd LLM Feedback Evaluation Framework, and provides the dataset, along with tools for parsing annotated PDF and DOCX files, a pipeline for generating LLM-based feedback on student essays, and UniMatch, a reference-based method for evaluating LLM-generated feedback. For full details, please refer to the paper.

## Download

Clone the repository:

```bash
git clone https://github.com/ShayanPey/SEFORA.git
cd SEFORA
```

---

## Dataset
SEFORA dataset consists of **564** student-written essay drafts, comprising 8,186 paragraphs. It includes 5,684 inline instructor annotations, with an **average of 147 words** of inline feedback per draft, as well as an additional **100 words per draft** in overall assessment comments.

Inline annotations are extracted from the annotated PDF/DOCX files and preserve original highlighting, along with associated comments, sticky notes, and strikeouts. The parser used for this extraction is provided in this repository.

In addition to inline feedback, the dataset includes overall instructor evaluations, consisting of rubric-based grades and summary comments provided at the document level.

### Direcotry Structure

The directory structure is hierarchical, with multiple nested subdirectories organizing the dataset:
```bash
SEFORA/
├── Batch_*/
│   ├── Course_*/
│   │   ├── Class_*/
│   │   │   ├── Essay_*/
│   │   │   │   ├── <Stage>/
│   │   │   │   │   ├── *.json
│   │   │   │   ├── *.txt
```
+ **Stage**: Draft stage of the essay (e.g., Outline, Revision, Final). Stage names are not standardized and reflect instructor's workflows.
+ **JSON files**: Parsed annotated essays.
+ **TXT files**: Assignment prompts and rubrics. (`.txt` files are mostly located within each `Essay_*` directory, though in some cases stage-specific or class-specific prompts or rubrics is provided within corresponding subdirectories.)
+ **File names**: Each `.json` file follows the naming structure `Batch_<#>_Course_<#>_Class_<#>_Essay_<#>_<Stage>_ID<XXXX>.json`. The dataset can be flattened into a single directory without any conflicts.

If you prefer to work with all JSON files in a single directory, you can flatten the structure using the following command:

```bash
mkdir -p SEFORA_flat && find SEFORA -type f -name "*.json" -exec cp {} SEFORA_flat/ \;
```

### JSON Structure
```json
{
  "paragraphs": [
    {
      "body": "<paragraph text>",
      "annotations": [
        {
          "type": "highlight | note | strikeout",
          "context_left": "<left context>",
          "context_right": "<right context>",
          "comment": "<instructor comment>",
          "text": "<highlighted text (only for type='highlight')>",
          "color": "green | pink | yellow | generic (only for type='highlight')"
        },
        ...
      ]
    },
    ...
  ],
  "annotations": [
    {
      "grades": {
        "rubric item title": 0.0,
        ...
      },
      "comment": "<instructor's overall comment/assessment>"
    }
  ]
}
```
---

## Generating LLM Feedback Pipeline
A notebook is provided for generating LLM feedback on student essays, along with a Python file for simpler job submission. The pipeline goes as follows:
+ Initiate LLM
+ Choose prompt template (Zero/few shot, guided/unguided, direct/CoT, prompt variant)
+ For every file in the dataset
  + For every paragraph with at least one annotation (ground truth) generate a feedback

## Evaluating LLM feedback (UniMatch)
UniMatch, as described in our paper, is a reference-based evaluation framework. It provides metrics such as precision, recall, and F1 score. UniMatch has two main components:
1. Segmentation: Segments the output of the LLM into "feedback units" (as described in the paper)
2. Similarity Matching: Computes pairwise semantic similarity between LLM-generated segments and gold units (instructor feedback)
At the end, a bipartite graph of pairwise scores is constructed. The Hungarian matching algorithm is then used to find the maximum matching that maximizes the total similarity, and soft precision, recall, and F1 scores are computed.

To reduce API costs, the second stage uses a buffering approach and is split into two steps:
1. Making the buffers and API calls
2. Retrieving results and running the maximum matching algorithm

All components are available both in the notebook and as separate Python files (one per component) to facilitate job submission.

---

## Parsers

### PDF Parser
PDFs are made of positioned elements and their coordinates in the page, which makes them highly portable but hard to parse. Refer to the source code for implementation details. Overview of the logic is as follows:
+ Pages are processed as collections of text blocks with their coordinates
+ For each block:
  + Extracts the textual content
  + Identifies highlight regions (quads) within the block and extracts the corresponding text
    + Groups quads to determine whether they belong to the same highlight
  + For each sticky note with coordinates being inside the text block, finds the left and right context of it
+ Injects the annotations
+ Reconstructs paragraphs based on geometric and layout cues

### DOCX Parser
Unlike PDFs, DOCX files are much easier to parse. DOCX file is basically a ZIP archive of XML documents, where `word/document.xml` contains the main text. The content is already organized into paragraphs (`<w:p>`), each composed of runs (`<w:r>`) representing formatted text spans.

---

## Citation
\[This is yet to be completed\]
If you use this dataset or code, please cite:

```bibtex
@article{sefora2026,
  title={...},
  author={...},
  journal={...},
  year={...}
}
```

---

## License

This dataset and code is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

You are free to share and adapt the material, provided that appropriate credit is given.

---

## Contact

For questions or issues:

* shayan.p@pitt.edu
