PROJECT CONTEXT — GECKO MED AI/ML ASSIGNMENT
=============================================

We are building an AI/ML engineering solution for Gecko Med as part of an
AI/ML Engineer interview assignment.

There are TWO independent assignments:

1. Question Paper Extraction Pipeline
2. Outlet Verification System

This workspace should contain both projects, but we are starting with the
QUESTION PAPER EXTRACTION PIPELINE.

==================================================
ASSIGNMENT 1 — QUESTION PAPER EXTRACTION
==================================================

Problem:

Universities publish examination question papers in different formats,
including PDFs, scanned documents, and images. The papers contain questions
and examination information, but their layout and structure can vary.

Build a pipeline that:

1. Processes uploaded question-paper PDFs.
2. Extracts individual questions.
3. Extracts examination metadata:
   - University name
   - Exam month
   - Exam year
   - Session code
4. Categorizes every question into one of:
   - Essay
   - Short Notes
   - Very Short Answers
   - MCQ

IMPORTANT DATA CHARACTERISTICS
--------------------------------

The provided Gecko Med dataset consists of PDFs.

The PDFs may:

- Contain scanned documents/images.
- Contain multiple question papers inside a single PDF.
- Contain many pages, potentially 50–100+ pages.
- Have different layouts between question papers.
- Contain seals/stamps.
- Contain scribblings or handwritten marks.
- Contain underlines/highlights.
- Contain poorly scanned or partially captured pages.
- Contain unwanted visual information that should not become extracted text.
- Have incomplete or unclear information.

Therefore, DO NOT assume:

PDF = one question paper.

A single PDF may contain multiple separate question papers.

The system must identify individual question papers within the document.

==================================================
CORE DESIGN PRINCIPLE
==================================================

This should be a robust DOCUMENT AI / OCR / INFORMATION EXTRACTION
pipeline.

Do NOT simply send the entire PDF to an LLM.

Do NOT assume every page is clean.

Do NOT hallucinate missing information.

If information cannot be reliably extracted, the system should indicate
uncertainty or mark the field/question for review.

The pipeline should degrade gracefully rather than producing incorrect
information.

==================================================
HIGH-LEVEL PIPELINE
==================================================

INPUT
  ↓
PDF upload
  ↓
Document ingestion
  ↓
Page extraction
  ↓
Page preprocessing
  ↓
Text extraction / OCR
  ↓
Noise filtering
  ↓
Layout analysis
  ↓
Question-paper boundary detection
  ↓
Metadata extraction
  ↓
Question segmentation
  ↓
Question classification
  ↓
Validation + confidence
  ↓
Structured JSON
  ↓
Frontend display

==================================================
DOCUMENT INGESTION
==================================================

The system should determine whether each page contains:

- Native digital text
- Scanned image
- Mixed content

For digital PDFs:
Use direct text extraction where reliable.

For scanned/image pages:
Use OCR.

Potential technologies to evaluate:

- PyMuPDF
- PaddleOCR
- OpenCV
- Pillow

Do not blindly OCR native text PDFs if direct extraction is more reliable.

==================================================
IMAGE / PAGE PREPROCESSING
==================================================

Investigate preprocessing techniques for:

- Deskewing
- Denoising
- Contrast enhancement
- Resolution normalization
- Rotation detection
- Cropping unwanted margins
- Removing obvious scan artifacts

The goal is to improve OCR quality without destroying meaningful content.

==================================================
NOISE HANDLING
==================================================

The source documents may contain:

- Scribbles
- Underlines
- Highlights
- Handwritten annotations
- Seals
- Stamps
- Watermarks
- Page numbers
- Repeated headers
- Repeated footers

The system should attempt to distinguish meaningful printed question text
from irrelevant visual noise.

Do NOT simply delete all non-text pixels.

Use OCR bounding boxes, confidence, layout information, and image
processing where appropriate.

==================================================
MULTIPLE QUESTION PAPERS
==================================================

This is a major requirement.

A single 50–100 page PDF may contain multiple question papers.

The system should identify paper boundaries using signals such as:

- Repeated examination header patterns
- University name
- Subject/course information
- Exam month/year
- Session code
- Question numbering reset
- Page structure/layout changes
- Section patterns
- Other document-level signals

Do not rely on a single hard-coded rule.

==================================================
METADATA EXTRACTION
==================================================

For each detected question paper, extract:

- University name
- Exam month
- Exam year
- Session code

Metadata may appear in different locations/layouts.

The extraction system should support variation.

If metadata is unreadable or absent:

Return null/unknown rather than inventing it.

==================================================
QUESTION EXTRACTION
==================================================

After identifying an individual question paper:

1. Detect question sections.
2. Detect question numbers.
3. Extract question text.
4. Preserve MCQ options when present.
5. Avoid including unrelated headers/footers/seals/scribblings.
6. Handle questions spanning multiple lines.
7. Handle questions spanning multiple pages where possible.

==================================================
QUESTION CLASSIFICATION
==================================================

Every extracted question must be classified into exactly one of:

- Essay
- Short Notes
- Very Short Answers
- MCQ

The classification approach should be selected based on the actual dataset.

Possible approaches:

- Rule-based classification
- Traditional ML
- Lightweight NLP model
- Transformer classifier

Do not introduce an LLM unless there is a demonstrated need.

==================================================
LLM / RAG
==================================================

RAG is NOT required.

The primary system should NOT depend on RAG.

An LLM is also NOT mandatory.

Prefer deterministic/document-AI/ML approaches where they are sufficient.

An LLM may be considered only as an optional fallback for ambiguous cases
if testing shows that it provides meaningful improvement.

==================================================
CONFIDENCE / VALIDATION
==================================================

The system should not blindly trust extraction.

Consider confidence for:

- OCR
- Metadata extraction
- Question segmentation
- Question classification

If information is insufficient:

Use something like:

status = "needs_review"

rather than guessing.

Example:

{
  "university": null,
  "confidence": 0.31,
  "status": "needs_review"
}

==================================================
EXPECTED OUTPUT
==================================================

The exact output format has NOT been specified by Gecko Med.

However, design the backend around structured JSON.

Expected conceptual structure:

{
  "document": "...",
  "question_papers": [
    {
      "university": "...",
      "exam_month": "...",
      "exam_year": "...",
      "session_code": "...",
      "questions": [
        {
          "number": "1",
          "text": "...",
          "type": "Essay",
          "confidence": 0.94
        }
      ]
    }
  ]
}

The exact schema should be refined after inspecting the provided dataset.

==================================================
FRONTEND
==================================================

Build a simple professional frontend.

User flow:

1. Upload PDF.
2. Show upload/processing status.
3. Process document.
4. Show detected question papers.
5. Show metadata.
6. Show extracted questions.
7. Show question classifications.
8. Show confidence/review warnings where applicable.
9. Provide a "View JSON" option.

Do not over-design the UI.

The focus is the AI/ML pipeline.

==================================================
BACKEND
==================================================

Preferred backend:

Python + FastAPI.

Potential structure:

backend/
  app/
    api/
    core/
    models/
    services/
      ingestion/
      preprocessing/
      ocr/
      layout/
      segmentation/
      extraction/
      classification/
      validation/
    main.py

The architecture should remain modular.

Do not put the entire pipeline into one Python file.

==================================================
PROCESSING LONG DOCUMENTS
==================================================

The system must support long PDFs.

Do NOT send a 100-page document as one huge model request.

Process pages/batches.

Prefer an asynchronous/background processing architecture.

Conceptually:

Upload
  ↓
Create Job ID
  ↓
Background processing
  ↓
Page processing
  ↓
Question-paper segmentation
  ↓
Extraction
  ↓
Classification
  ↓
Validation
  ↓
Completed result

==================================================
TECHNOLOGY DIRECTION
==================================================

Initial technologies to investigate:

Backend:
- Python
- FastAPI

PDF:
- PyMuPDF

Image:
- OpenCV
- Pillow

OCR:
- PaddleOCR
- Tesseract as an alternative if useful

ML/NLP:
- scikit-learn
- sentence-transformers / transformer models if required

Validation:
- Pydantic

Frontend:
- React
- TypeScript

Deployment:
- Docker
- Render or another suitable hosting platform

Do not install/use every technology blindly.

First inspect the dataset and determine what is actually necessary.

==================================================
IMPORTANT ENGINEERING REQUIREMENT
==================================================

Before implementation:

1. Inspect the provided question-paper PDFs.
2. Determine:
   - Number of PDFs
   - Number of pages
   - Native vs scanned pages
   - Number of question papers per PDF
   - Common headers
   - Metadata patterns
   - Question numbering patterns
   - Different layouts
   - OCR difficulty
   - Presence of seals/scribbles/annotations
   - MCQ formatting
   - Section formatting
3. Based on this analysis, propose the final architecture.
4. Identify the minimum required dependencies.
5. Create a small prototype on representative pages.
6. Evaluate the prototype.
7. Only then implement the complete pipeline.

DO NOT prematurely over-engineer.

==================================================
INTERVIEW CONTEXT
==================================================

This is an AI/ML Engineer interview assignment for Gecko Med.

The goal is not merely to demonstrate that an OCR library works.

The solution should demonstrate:

- Problem understanding
- AI/ML engineering
- Document processing
- Robustness
- Appropriate model/tool selection
- Handling messy real-world inputs
- Structured outputs
- Confidence/uncertainty handling
- API engineering
- Production-oriented architecture
- Evaluation

The solution should be explainable during a technical discussion.

==================================================
CURRENT TASK
==================================================

Do NOT immediately start writing the complete application.

First:

1. Inspect the provided dataset.
2. Analyze representative PDFs/pages.
3. Identify document patterns and edge cases.
4. Propose the concrete architecture.
5. Propose the technology/model choices with reasons.
6. Propose the JSON schema.
7. Propose an evaluation strategy.
8. Wait for review/approval before implementing the full pipeline.

The system should be built based on evidence from the actual Gecko Med
dataset rather than assumptions.