Banking SLM Pipeline
First things first: Before writing any code, set up your AI tool to log all interactions to a prompt-log.md file in the root of your repository. This file is required — it will be reviewed as part of your evaluation. Every prompt you send and every response you receive should be captured. Start now.

Scenario
You are an ML engineer at a small banking startup building an internal knowledge assistant. Relationship managers and analysts frequently field questions about financial concepts, products, and markets. Leadership wants a fine-tuned small language model that can answer financial questions accurately — grounded in real financial knowledge, not hallucinated.

The company has no ML infrastructure yet. There is no GPU cluster, no feature store, no experiment tracker. You are standing this up from scratch on a laptop. The model must be small enough to fine-tune and run inference on CPU — the first deployment target is a single-container API.

Your job: build an end-to-end fine-tuning pipeline that takes a pretrained small language model, adapts it to the financial domain using publicly available instruction data, evaluates it rigorously, and serves it behind an API.

Data Reference
Your training dataset is Finance Alpaca, a publicly available financial instruction-tuning dataset on Hugging Face:

Dataset: gbharti/finance-alpaca

~68,000 instruction/input/output examples covering financial topics
Loaded via the datasets library: load_dataset("gbharti/finance-alpaca")
The dataset has no predefined splits — you will need to create your own train/validation/test split
Each example has instruction, input, and output fields in Alpaca format
A few things to keep in mind:

The dataset is too large to train on entirely within your time window. You will need to make deliberate decisions about how to sample, filter, or curate a training subset. Document your strategy and rationale.
The data quality is uneven. Some examples are high-quality regulatory Q&A, others are generic financial trivia, and some are low-quality or near-duplicates. Not all examples are equally useful for a banking knowledge assistant.
The input field is sometimes empty, sometimes contains context. Your preprocessing needs to handle both cases.
Do not fabricate evaluation results or hard-code outputs. Your pipeline must actually train and evaluate.
You may also explore sujet-ai/Sujet-Finance-Instruct-177k (177K examples from 18 finance datasets) as a supplementary or alternative source, but manage your time — more data is not always better.

Model Reference
You must fine-tune a pretrained model from the Hugging Face Hub. You may choose your own, but here are models known to be feasible on CPU:

Model	Params	Notes
Qwen/Qwen2.5-0.5B	0.5B	Good balance of capability and speed on CPU
HuggingFaceTB/SmolLM-135M	135M	Very fast to train, lower quality outputs
Larger models (1B+) will likely be too slow for CPU training in your time window. A model that finishes training is worth more than one that doesn't.

Requirements
Tier 1 — Working Pipeline (Required)
Build an end-to-end pipeline that fine-tunes and serves a financial QA model. A reviewer should be able to clone your repo, install dependencies, and reproduce your results by following your README — without reading your code first.

Data Preparation — Load the dataset from Hugging Face and build a training pipeline:

Create a train/validation/test split
Sample or curate a training subset appropriate for your time and compute constraints — document your selection strategy
Format examples into a prompt template appropriate for your chosen model — the template you train on should match how you intend to query the model at inference time. Justify your template design in the README.
Handle variable data quality (missing fields, inconsistent formatting)
Model Fine-Tuning — Fine-tune a pretrained model on the financial QA task. Your training script must:

Choose an appropriate fine-tuning approach for your model size and compute constraints (parameter-efficient fine-tuning, full fine-tuning, or otherwise) — justify your choice in the README
Accept configurable hyperparameters (learning rate, batch size, epochs, and any method-specific parameters) without requiring code changes — via config file, CLI args, or environment variables
Set random seeds for reproducibility
Log training loss and validation metrics to stdout or a log file
Save the adapted model to disk in a format that can be loaded for inference without retraining
Evaluation — Evaluate the fine-tuned model on your held-out test set. At minimum produce:

Quantitative metrics appropriate for generative QA (e.g., ROUGE, BERTScore, or semantic similarity against reference answers) — and an acknowledgment of what these metrics do and don't capture
A qualitative sample: pick 10–20 test examples spanning different question types and show the model's actual outputs alongside the reference answers
An honest assessment of where the model performs well and where it fails
Inference API — Serve the fine-tuned model behind a REST API:

POST /ask — accepts a JSON body with a question field (and optional context field), returns the model's generated answer
GET /health — returns a 200 with model metadata (base model name, adaptation method, training date)
The API must load the model from disk at startup (not retrain on every boot).

A polished Tier 1 with honest evaluation and a working API is a strong submission. A rushed attempt at everything is not.

Tier 2 — Production Thinking (Expected)
Experiment Tracking — Implement structured experiment tracking so that training runs are reproducible and comparable. For each run, persist:

All hyperparameters and configuration
Per-step metrics (training loss, validation metrics, test metrics)
A way to identify which saved model corresponds to which run
Upstream data lineage (what dataset version and subset was used)
You can use MLflow (local), Weights & Biases (local mode), a structured JSON log, or any other approach — but it must be queryable after the fact.

Refusal & Guardrails — A financial assistant that confidently hallucinates wrong answers is worse than one that says "I don't know." Mix unanswerable or out-of-scope questions into your evaluation set and measure how the model handles them. Does it refuse gracefully, or does it fabricate an answer? Describe your refusal strategy and the tradeoffs you expect in the README.

Tier 3 — Depth (Differentiators)
Pick any that interest you — depth on one is better than surface-level on all:

Evaluation Framework Design — Standard text generation metrics (ROUGE, BLEU) only scratch the surface for a financial knowledge assistant. Design and implement an evaluation approach that goes beyond surface-level similarity. What properties matter when a banker relies on this model's answers? How would you measure them? Implement what you can within the time window and describe what you would add with more time.

Data Curation & Quality — Build a scoring pipeline that rates training examples by relevance and quality. Train the model on your curated subset and on a naive random sample of the same size. Does curation measurably help? Present the comparison.

Multi-Run Comparison — Train the model under at least two meaningfully different configurations (different adaptation parameters, different data subsets, different prompt formats) and produce a structured comparison. Which choices mattered most? Present results in a way that would help a team decide which configuration to deploy.

Deliverables
Push your code to a public GitHub repository
Your solution must be written in Python
Single-command startup for both training and serving (make train, make serve, docker compose up, or equivalent)
Include a README.md with:
Architecture overview and key design decisions
Setup and run instructions (including Python version and dependency install)
How to reproduce training results
Model performance summary and honest assessment of limitations
API documentation with example curl commands
What you would do differently with more time
Include your prompt-log.md capturing all AI tool interactions (required)
Your pipeline and API should be runnable and testable locally on CPU
Evaluation
We evaluate both what you built and how you built it (via your prompt log and commit history). Building with AI is required and expected — what matters is how effectively you direct it.