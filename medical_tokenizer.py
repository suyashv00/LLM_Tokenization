"""
Medical BPE Tokenizer
=====================
BPE tokenizer trained on PubMed medical text.
Includes pre-tokenization, normalization, training, encoding, and decoding.
"""

import re
import os
import heapq
import unicodedata
import pickle
import json
import time
import tracemalloc
from collections import defaultdict

import regex
from tqdm import tqdm


# --- Pre-tokenization Regexes ------------------------------------------------

GPT2_PRE_TOKEN_REGEX = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Medical regex: keeps IL-6, COVID-19, p53, BRCA1, HbA1c, 25mg, 0.05 intact
MEDICAL_PRE_TOKEN_REGEX = (
    r"""(?:\d+[\-\.]\d+(?:[a-zA-Z]+)?)|"""
    r"""(?:[A-Za-z]+\d+[A-Za-z]*)|"""
    r"""(?:[A-Z]{2,}(?:[-]\d+)+)|"""
    r"""(?:\d+(?:\.\d+)?\s?(?:mg|mL|ug|kg|mmol|umol|nmol|IU|mM|nM))|"""
    r"""'(?:[sdmt]|ll|ve|re)|"""
    r""" ?\p{L}+(?:[-]\p{L}+)*|"""
    r""" ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|"""
    r"""\s+(?!\S)|\s+"""
)

MEDICAL_SPECIAL_TOKENS = ["<|endoftext|>"]

# Text Normalization

_GREEK_MAP = {
    "α": "alpha", "β": "beta",  "γ": "gamma",  "δ": "delta",
    "ε": "epsilon","ζ": "zeta", "η": "eta",    "θ": "theta",
    "κ": "kappa",  "λ": "lambda","μ": "u",     "ν": "nu",
    "ξ": "xi",     "π": "pi",   "ρ": "rho",   "σ": "sigma",
    "τ": "tau",    "φ": "phi",  "χ": "chi",   "ψ": "psi",
    "ω": "omega",  "Α": "Alpha","Β": "Beta",  "Γ": "Gamma",
    "Δ": "Delta",  "Κ": "Kappa","Λ": "Lambda","Π": "Pi",
    "Σ": "Sigma",  "Ω": "Omega",
}

_SYMBOL_MAP = {
    "±": "+/-",  "≥": ">=",    "≤": "<=",   "×": "x",
    "÷": "/",    "·": ".",     "−": "-",    "–": "-",
    "—": "-",    "\u00b0": " degrees ", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
}


def normalize_medical_text(text: str) -> str:
    """
    Normalize PubMed text before BPE training.
    """
    text = unicodedata.normalize("NFC", text)
    for greek, ascii_eq in _GREEK_MAP.items():
        text = text.replace(greek, ascii_eq)
    for symbol, repl in _SYMBOL_MAP.items():
        text = text.replace(symbol, repl)
    text = regex.sub(r"\\[a-zA-Z]+", "", text)          # LaTeX
    text = (text.replace("&lt;", "<").replace("&gt;", ">")
                .replace("&amp;", "&").replace("&nbsp;", " ")
                .replace("&#160;", " "))                 # HTML entities
    text = regex.sub(r" {2,}", " ", text)
    text = regex.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_medical_text(text: str, min_length: int = 100) -> bool:
    """
    Filter out DNA sequences, junk data, and non-medical content.
    """
    if len(text) < min_length:
        return False
    
    # Reject DNA sequences (ATCG patterns)
    dna_ratio = len(re.findall(r'[ATCG]', text)) / len(text)
    if dna_ratio > 0.5:  # >50% DNA bases = likely sequence
        return False
    
    # Reject protein sequences (long amino acid strings)
    protein_chars = set('ACDEFGHIKLMNPQRSTVWY')
    if sum(1 for c in text if c in protein_chars) / len(text) > 0.6:
        return False

    # Accept if medical keywords
    medical_keywords = {
        'patient', 'treatment', 'disease', 'clinical', 'study',
        'diagnosis', 'therapy', 'medical', 'hospital', 'doctor',
        'symptom', 'drug', 'medicine', 'cancer', 'heart', 'brain',
        'infection', 'surgery', 'outcome', 'method', 'result',
        'abstract', 'conclusion', 'background', 'objective',
    }
    text_lower = text.lower()
    keyword_count = sum(1 for kw in medical_keywords if kw in text_lower)
    
    return keyword_count >= 2  # least 2 keywords


def create_filtered_medical_corpus(
    dataset,
    output_path: str,
    text_field: str = "article",
    normalize: bool = True,
    boundary_token: str = "<|endoftext|>",
    sample_percent: float = 100,
) -> None:
    
    written = 0
    skipped = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, desc="Writing corpus"):
            text = item[text_field]
            
            # Filter step
            if not is_medical_text(text):
                skipped += 1
                continue
            
            if normalize:
                text = normalize_medical_text(text)
            
            if text.strip():
                f.write(text)
                f.write(f"\n{boundary_token}\n")
                written += 1


class ReversedBytes:
    """
    Reverses byte comparison so Python's min-heap surfaces
    lexicographically LARGEST bytes on frequency ties.
    """
    __slots__ = ("data",)

    def __init__(self, data: bytes):
        self.data = data

    def __lt__(self, other):
        return self.data > other.data

    def __le__(self, other):
        return self.data >= other.data

    def __eq__(self, other):
        return isinstance(other, ReversedBytes) and self.data == other.data


def pretokenize(
    chunk: str,
    special_tokens: list[str],
    use_medical_regex: bool = True,
) -> dict[tuple[int, ...], int]:
    """
    Pretokenize a chunk into byte-tuple → frequency mapping.
    Splits on special tokens first so they never enter the regex.
    """
    pattern = MEDICAL_PRE_TOKEN_REGEX if use_medical_regex else GPT2_PRE_TOKEN_REGEX
    special_set = set(special_tokens)

    if special_tokens:
        escaped = [regex.escape(st) for st in special_tokens]
        parts = regex.split(f"({'|'.join(escaped)})", chunk)
        text_parts = [p for p in parts if p not in special_set and p]
    else:
        text_parts = [chunk]

    freqs: dict[tuple[int, ...], int] = {}
    for part in text_parts:
        for m in regex.finditer(pattern, part):
            key = tuple(m.group().encode("utf-8"))
            freqs[key] = freqs.get(key, 0) + 1
    return freqs


def train_medical_bpe_tokenizer(
    input_path: str,
    vocab_size: int = 32_000,
    special_tokens: list[str] | None = None,
    use_medical_regex: bool = True,
    heap_compaction_ratio: float = 3.0,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer for medical/PubMed text.
    """
    if special_tokens is None:
        special_tokens = MEDICAL_SPECIAL_TOKENS

    corpus_size = os.path.getsize(input_path)
    print(f"Corpus: {corpus_size / 1024 / 1024:.1f} MB")

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        chunk = f.read()
    word_freq_raw = pretokenize(chunk, special_tokens, use_medical_regex)

    # ── Phase 2: Initialize Vocab & Structures ─────────────────────────
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    words: dict[int, list[int]] = {}
    word_freqs: dict[int, int] = {}
    for wid, (wt, freq) in enumerate(word_freq_raw.items()):
        words[wid] = list(wt)
        word_freqs[wid] = freq

    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        raise ValueError(
            f"vocab_size={vocab_size} must exceed initial size={len(vocab)} "
            f"(256 bytes + {len(special_tokens)} special tokens)"
        )
    merges: list[tuple[bytes, bytes]] = []
    print(f"Initial vocab: {len(vocab)} | Merges to do: {num_merges:,}")

    # ── Phase 3: Pair Frequencies + Inverted Index ─────────────────────
    pair_frequencies: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)

    for wid, tokens in words.items():
        freq = word_freqs[wid]
        for i in range(len(tokens) - 1):
            p = (tokens[i], tokens[i + 1])
            pair_frequencies[p] += freq
            pair_to_words[p].add(wid)

    print(f"Unique pairs: {len(pair_frequencies):,}")

    # ── Phase 4: Build Heap ────────────────────────────────────────────
    def make_entry(pair):
        freq = pair_frequencies[pair]
        lex = (ReversedBytes(vocab[pair[0]]), ReversedBytes(vocab[pair[1]]))
        return (-freq, lex, pair)

    heap = [make_entry(p) for p in pair_frequencies]
    heapq.heapify(heap)
    valid_pairs: set[tuple[int, int]] = set(pair_frequencies.keys())

    # ── Phase 5: Merge Loop ────────────────────────────────────────────
    for _ in tqdm(range(num_merges), desc="BPE merges", unit="merge"):

        # Find best valid pair (lazy deletion)
        best_pair = None
        while heap:
            neg_freq, _lex, pair = heapq.heappop(heap)
            if pair not in valid_pairs:
                continue
            cur_freq = pair_frequencies.get(pair, 0)
            if cur_freq != -neg_freq:
                if cur_freq > 0:
                    heapq.heappush(heap, make_entry(pair))
                continue
            best_pair = pair
            break

        if best_pair is None:
            print("No more pairs — stopping early.")
            break

        # Create merged token
        new_id = len(vocab)
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # Update only affected words via inverted index
        for wid in list(pair_to_words.get(best_pair, set())):
            tokens = words[wid]
            freq = word_freqs[wid]

            # A: Remove old pair counts
            for i in range(len(tokens) - 1):
                p = (tokens[i], tokens[i + 1])
                pair_frequencies[p] -= freq
                if pair_frequencies[p] <= 0:
                    del pair_frequencies[p]
                    valid_pairs.discard(p)
                pair_to_words[p].discard(wid)
                # FIX 1: clean up empty sets → prevents memory leak
                if p in pair_to_words and not pair_to_words[p]:
                    del pair_to_words[p]

            # B: Apply merge
            new_tokens, i = [], 0
            while i < len(tokens):
                if (i < len(tokens) - 1
                        and tokens[i] == best_pair[0]
                        and tokens[i + 1] == best_pair[1]):
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            words[wid] = new_tokens

            # C: Add new pair counts
            for i in range(len(new_tokens) - 1):
                p = (new_tokens[i], new_tokens[i + 1])
                pair_frequencies[p] += freq
                valid_pairs.add(p)
                pair_to_words[p].add(wid)
                heapq.heappush(heap, make_entry(p))

        # Clean up merged pair
        valid_pairs.discard(best_pair)
        # FIX 2: explicit pop prevents pair_frequencies/valid_pairs desync
        pair_frequencies.pop(best_pair, None)
        if best_pair in pair_to_words:
            del pair_to_words[best_pair]

        # Heap compaction — keep stale entries under control
        if len(heap) > heap_compaction_ratio * len(valid_pairs):
            heap = [make_entry(p) for p in valid_pairs if p in pair_frequencies]
            heapq.heapify(heap)

    return vocab, merges


class MedicalBPETokenizer:
    """
    BPE tokenizer for medical text.
    Loads vocab and merges from JSON/pickle files produced by train_medical_bpe_tokenizer.

    Usage:
        tokenizer = MedicalBPETokenizer("./results")
        ids  = tokenizer.encode("Patient treated with IL-6 inhibitors.")
        text = tokenizer.decode(ids)
    """

    def __init__(self, results_dir: str, use_medical_regex: bool = True):
        self.pattern = MEDICAL_PRE_TOKEN_REGEX if use_medical_regex else GPT2_PRE_TOKEN_REGEX

        # Load from pickle files
        with open(f"{results_dir}/vocab.pkl", "rb") as f:
            self.vocab: dict[int, bytes] = pickle.load(f)
        
        with open(f"{results_dir}/merges.pkl", "rb") as f:
            merges: list[tuple[bytes, bytes]] = pickle.load(f)

        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # merge_rank[(id_a, id_b)] = priority  (lower = applied first)
        self.merge_rank: dict[tuple[int, int], int] = {}
        for rank, (a_bytes, b_bytes) in enumerate(merges):
            if a_bytes in self.bytes_to_id and b_bytes in self.bytes_to_id:
                pair = (self.bytes_to_id[a_bytes], self.bytes_to_id[b_bytes])
                self.merge_rank[pair] = rank
        print(f"Loaded : {len(self.vocab):,} tokens | {len(self.merge_rank):,} merges")


    def _apply_merges(self, token_ids: list[int]) -> list[int]:
        """
        Apply merges in priority order (lowest rank first).
        Uses an iterative approach: repeatedly find & apply the highest-priority merge.
        """
        while len(token_ids) >= 2:
            # Find the pair with the lowest merge_rank (highest priority)
            best_rank = float("inf")
            best_idx = -1
            best_pair = None
            
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                rank = self.merge_rank.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
                    best_pair = pair
            
            # No more pairs to merge
            if best_idx == -1:
                break
            
            # Apply the best merge
            a_id = token_ids[best_idx]
            b_id = token_ids[best_idx + 1]
            
            # Look up the merged token ID
            merged_bytes = self.vocab[a_id] + self.vocab[b_id]
            merged_id = self.bytes_to_id.get(merged_bytes)
            
            if merged_id is None:
                # This pair exists in merge_rank but not in vocab — shouldn't happen
                del self.merge_rank[best_pair]
                continue
            
            # Replace the pair with the merged token
            token_ids = token_ids[:best_idx] + [merged_id] + token_ids[best_idx + 2:]
        
        return token_ids
    

    def encode(self, text: str, normalize: bool = True) -> list[int]:
        """
        Encode text to token IDs.
        Matches the training pretokenization behavior (spaces are significant).
        """
        if normalize:
            text = normalize_medical_text(text)
        
        token_ids = []
        for match in regex.finditer(self.pattern, text):
            word_str = match.group()
            word_bytes = word_str.encode("utf-8")
            
            # OPTIMIZATION: Check if the entire word is already in vocab as a single token
            # (This handles merged words from training)
            if word_bytes in self.bytes_to_id:
                token_ids.append(self.bytes_to_id[word_bytes])
            else:
                # Fall back to byte-level merging
                word_ids = [self.bytes_to_id[bytes([b])] for b in word_bytes]
                token_ids.extend(self._apply_merges(word_ids))
        
        return token_ids
    

    def decode(self, token_ids: list[int]) -> str:
        chunks = [self.vocab[tid] for tid in token_ids if tid in self.vocab]
        return b"".join(chunks).decode("utf-8", errors="replace")
    

    def vocab_size(self) -> int:
        return len(self.vocab)
