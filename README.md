# 🧠 Synthetic Data Kit

> 🛠 **Built by Meta AI** — All credit to [Meta](https://github.com/meta-llama/synthetic-data-kit) for creating this tool to help generate synthetic datasets for fine-tuning LLMs.

Generate Reasoning Traces, QA Pairs, summaries, and convert them into formats ready for instruction tuning — all with a simple CLI interface.

📘 [Guide: Adding Reasoning to LLaMA 3](https://github.com/meta-llama/synthetic-data-kit/tree/main/use-cases/adding_reasoning_to_llama_3)

---

## 🔧 About This Fork / Custom Setup

This README and setup have been **modified to run locally using [Ollama](https://ollama.com/)** instead of vLLM or external APIs.

---

## 🚀 Quick Start (Ollama Setup)

### Step 1: Clone this repository

```bash
git clone https://github.com/shreyanshjain05/synthetic-data-kit-with-ollama.git
cd synthetic-data-kit-with-ollama
````

### Step 2: Create and activate a virtual environment

```bash
# Using conda (recommended)
conda create -n synthetic-data python=3.10
conda activate synthetic-data

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -e .
```

---

## ❗ Why Use Ollama?

The original toolkit was configured for **LLaMA 3 70B**, served via `vLLM`, which requires a multi-GPU cloud setup and cannot run locally on most machines.

To enable **local fine-tuning dataset generation**, this version uses:

✅ `llama3:latest` via Ollama (a CPU/GPU-friendly runtime)
✅ No HuggingFace token, cloud API, or GPU cluster needed
✅ Easily runs on MacBooks, laptops, or small servers

---

## 🐙 Using Ollama (Local LLM Backend)

### Step 4: Install Ollama

Download from [https://ollama.com/download](https://ollama.com/download)

### Step 5: Pull the LLaMA 3 model

```bash
ollama pull llama3
```

This fetches `llama3:latest` (\~8B version, fits on local devices).

### Step 6: Run Ollama

```bash
ollama serve
```

This automatically starts the Ollama server at:

```
http://localhost:11434
```

---

## ⚙️ Configure `config.yaml` for Ollama

Edit or create your `configs/config.yaml` file like this:

```yaml
llm:
  provider: "api-endpoint"

api-endpoint:
  api_base: "http://localhost:11434/v1"
  api_key: "not-needed"
  model: "llama3:latest"

generation:
  temperature: 0.7
  chunk_size: 4000
  num_pairs: 25

curate:
  threshold: 8.0
  batch_size: 8
```

✅ Check Ollama is working:

```bash
synthetic-data-kit -c configs/config.yaml system-check        
```

---

## 🔁 Example Workflow (Now using Ollama)

```bash
# 1. Ingest a document
synthetic-data-kit ingest docs/report.pdf

# 2. Create QA pairs or Chain-of-Thought examples
synthetic-data-kit create data/parsed/report.txt --type qa -n 20

# 3. Curate examples using LLaMA 3 via Ollama
synthetic-data-kit curate data/generated/report_qa_pairs.json --threshold 8.0

# 4. Save in fine-tuning format
synthetic-data-kit save-as data/curated/report_cleaned.json --format alpaca
```

---

## 🧠 Original Meta Pipeline (Unmodified)

```bash
synthetic-data-kit ingest ...
synthetic-data-kit create ...
synthetic-data-kit curate ...
synthetic-data-kit save-as ...
```

Supports:

* PDF, DOCX, HTML, YouTube
* QA, Chain-of-Thought, summaries
* Formats: Alpaca, ChatML, HF, FT

---

## 🗂️ Process Multiple Files

```bash
synthetic-data-kit ingest ./documents/
synthetic-data-kit create ./data/parsed/ --type qa -n 30
synthetic-data-kit curate ./data/generated/ --threshold 8.5
synthetic-data-kit save-as ./data/curated/ --format ft --storage hf
```

---

## 🔧 Chunking for Long Documents

| Parameter       | Default | Description                   |
| --------------- | ------- | ----------------------------- |
| `chunk_size`    | 4000    | Text chunk size in characters |
| `chunk_overlap` | 200     | Overlap to preserve context   |

```bash
synthetic-data-kit create file.txt --chunk-size 3000 --chunk-overlap 100
```

---

## 📜 Configuration Summary

```yaml
llm:
  provider: "api-endpoint"

api-endpoint:
  api_base: "http://localhost:11434/v1"
  api_key: "not-needed"
  model: "llama3:latest"

generation:
  temperature: 0.7
  chunk_size: 4000
  num_pairs: 25

curate:
  threshold: 8.0
  batch_size: 8
```

Use with:

```bash
synthetic-data-kit -c configs/config.yaml ingest yourfile.pdf
```

---

## 🧩 Supported Formats

* ✅ PDF
* ✅ DOCX
* ✅ PPTX
* ✅ HTML
* ✅ TXT
* ✅ YouTube transcripts

---

## 🔍 Troubleshooting

| Issue                     | Solution                                                         |
| ------------------------- | ---------------------------------------------------------------- |
| Ollama not responding     | Make sure `ollama serve` is running                              |
| JSON errors during curate | Use `--verbose`; install `json5`; lower batch size               |
| Memory limits             | Reduce `num_pairs`, `chunk_size`, or `batch_size`                |
| Parsing errors            | Ensure correct parser libraries installed (`pdfminer.six`, etc.) |

---

## 📄 License

MIT License — see [LICENSE](./LICENSE)

---

```

---

Let me know if you want this pushed as a file, or want me to help generate a `config.yaml`, `.gitignore`, or helper script too.
```
