# 🧬 Medical BPE Tokenizer — From Scratch

A **Byte Pair Encoding (BPE)** tokenizer built from scratch, specifically designed for **biomedical and clinical text** using the [PubMed Summarization](https://huggingface.co/datasets/ccdv/pubmed-summarization) dataset.

> *"Tokenization is at the heart of much weirdness of LLMs. Do not brush it off."* — Andrej Karpathy

![Tokenization reference](reference.png)

---

## 🎯 Why a Medical Tokenizer?

General-purpose tokenizers (GPT-2, GPT-4) are trained on web text and **waste tokens** on medical terminology. A domain-adapted tokenizer means **better compression**, **fewer tokens per document**, and **more efficient LLM training** on medical data.

**Example:** The word "gastrointestinal" takes 3 tokens in GPT-2 but just 1 token in our medical tokenizer.

---

## ✨ Key Features

- **Medical-aware pretokenization regex** — preserves `IL-6`, `COVID-19`, `BRCA1`, `HbA1c`, `25mg`, `0.05` as atomic units
- **Greek letter & symbol normalization** — `α → alpha`, `β → beta`, `± → +/-`, `≥ → >=`
- **LaTeX & HTML entity cleanup** — strips LaTeX commands, decodes HTML entities
- **Data quality filtering** — rejects DNA/protein sequences, requires medical keyword presence
- **Heap-optimized BPE training** — lazy-deletion max-heap with compaction for efficient merges
- **Inverted index** (`pair_to_words`) — only updates affected words during each merge
- **Deterministic tie-breaking** — `ReversedBytes` ensures reproducible merge order on frequency ties

---

## 📁 Project Structure

```
├── tokenization.ipynb            # Main BPE pipeline: pretokenize → train → encode/decode
├── tokenization_types.ipynb      # Educational: word vs char vs subword tokenization
├── results/
│   ├── vocab.json                # 32,000-token vocabulary (human-readable)
│   ├── merges.json               # Ordered merge rules (human-readable)
│   ├── vocab.pkl                 # Vocab in pickle format (fast loading)
│   └── merges.pkl                # Merges in pickle format (fast loading)
├── input.txt                     # Sample text for prototyping
├── References.txt                # Learning resources & references
├── requirements.txt              # Python dependencies
└── reference.png                 # Karpathy's tokenization importance slide
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### 1. Generate the Corpus (one-time)

```python
from datasets import load_dataset

ds = load_dataset("ccdv/pubmed-summarization", split="train[:10%]")
create_filtered_medical_corpus(ds, "pubmed_filtered_corpus.txt")
```

This downloads ~10% of PubMed abstracts, filters out non-medical content (DNA sequences, junk), normalizes text, and writes a clean ~200 MB corpus.

### 2. Train the Tokenizer

```python
vocab, merges = train_medical_bpe_tokenizer(
    input_path="pubmed_filtered_corpus.txt",
    vocab_size=32_000,
    use_medical_regex=True,
)
```

### 3. Use the Tokenizer

```python
tokenizer = MedicalBPETokenizer("./results")

# Encode
ids = tokenizer.encode("The patient was treated with IL-6 inhibitors for rheumatoid arthritis.")
print(ids)  

# Decode 
text = tokenizer.decode(ids)
print(text)
```

---

## 🏗️ Architecture

### Pipeline Overview

```
PubMed Dataset
    │
    ▼
┌──────────────────────┐
│  Data Filtering      │  DNA/protein rejection, medical keyword gate
│  Text Normalization  │  Greek letters, symbols, LaTeX, HTML
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  Pretokenization     │  Medical-aware regex splitting
│  (byte-level)        │  Preserves IL-6, HbA1c, 25mg as units
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  BPE Training        │  Heap-based merge loop with inverted index
│  32,000 merges       │  Compaction ratio for memory efficiency
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  Tokenizer Class     │  encode() / decode() with merge ranking
│  (MedicalBPETokenizer)│ Lossless roundtrip guaranteed
└──────────────────────┘
```

### Medical Pretokenization Regex

The custom regex handles medical-specific patterns that general tokenizers break:

| Pattern | Example | What It Preserves |
|---|---|---|
| Alphanumeric IDs | `IL-6`, `p53`, `BRCA1` | Gene/protein names |
| Dosage units | `25mg`, `0.05mL` | Measurements |
| Hyphenated terms | `COVID-19`, `HbA1c` | Medical identifiers |
| Contractions | `don't`, `we'll` | Natural language |

---

## 📓 Educational Notebook

`tokenization_types.ipynb` demonstrates the three main tokenization approaches:

1. **Word-based** — simple whitespace/regex splitting
2. **Character-based** — individual character tokens
3. **Subword-based (BPE)** — the sweet spot between word and character level

It also includes a comparison with OpenAI's `tiktoken` library.

---

## 📊 Training Stats

| Metric | Value |
|---|---|
| Corpus size | ~200 MB (PubMed 10% split) |
| Initial vocab | 257 (256 bytes + 1 special token) |
| Final vocab | 32,000 tokens |
| Merges performed | 31,743 |

## 🔗 References & Acknowledgments

1. [Sebastian Raschka — BPE from Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)
2. [Building a Fast BPE Tokenizer from Scratch](https://jytan.net/blog/2025/bpe/)
3. [Andrej Karpathy — Let's build the GPT Tokenizer](https://youtu.be/fKd8s29e-l4?si=zOHCbc1fWFSZJneO)
4. [Karpathy's minbpe](https://github.com/karpathy/minbpe)
5. [Imad Dabbura — BPE Tokenizer](https://imaddabbura.github.io/posts/nlp/BPE-Tokenizer.html)