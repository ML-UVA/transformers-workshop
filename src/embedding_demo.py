#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def tokenize(text: str):
    return re.findall(r"[a-z']+", text.lower())


def load_corpus(path: Path):
    tokens = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            tokens.extend(tokenize(line))
    return tokens


def build_vocab(tokens, min_count):
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    vocab = [w for w, c in counts.items() if c >= min_count]
    vocab.sort(key=lambda w: (-counts[w], w))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    return vocab, word_to_id, counts


def build_cooccurrence(tokens, word_to_id, window):
    size = len(word_to_id)
    matrix = np.zeros((size, size), dtype=np.float64)
    for i, center in enumerate(tokens):
        c_id = word_to_id.get(center)
        if c_id is None:
            continue
        left = max(0, i - window)
        right = min(len(tokens), i + window + 1)
        for j in range(left, right):
            if i == j:
                continue
            w_id = word_to_id.get(tokens[j])
            if w_id is None:
                continue
            matrix[c_id, w_id] += 1.0
    return matrix


def ppmi_matrix(cooc):
    total = cooc.sum()
    row_sums = cooc.sum(axis=1)
    col_sums = cooc.sum(axis=0)
    size = cooc.shape[0]
    ppmi = np.zeros_like(cooc)
    for i in range(size):
        if row_sums[i] == 0:
            continue
        for j in range(size):
            if cooc[i, j] == 0 or col_sums[j] == 0:
                continue
            pmi = math.log((cooc[i, j] * total) / (row_sums[i] * col_sums[j]))
            if pmi > 0:
                ppmi[i, j] = pmi
    return ppmi


def train_embeddings(corpus_path, window, dim, min_count):
    tokens = load_corpus(corpus_path)
    vocab, word_to_id, counts = build_vocab(tokens, min_count)
    if len(vocab) == 0:
        raise ValueError("No vocabulary items meet min_count; try lowering --min-count.")
    cooc = build_cooccurrence(tokens, word_to_id, window)
    ppmi = ppmi_matrix(cooc)
    u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
    dim = min(dim, u.shape[1])
    embeddings = u[:, :dim] * np.sqrt(s[:dim])
    return vocab, embeddings, counts


def read_pretrained(path: Path):
    vectors = []
    vocab = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip().split()
        if len(first) == 2 and all(part.isdigit() for part in first):
            dim = int(first[1])
        else:
            word = first[0]
            vec = np.array([float(x) for x in first[1:]], dtype=np.float64)
            dim = vec.size
            if word not in seen:
                vocab.append(word)
                vectors.append(vec)
                seen.add(word)
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float64)
            if vec.size != dim:
                continue
            if word in seen:
                continue
            vocab.append(word)
            vectors.append(vec)
            seen.add(word)
    embeddings = np.vstack(vectors)
    return vocab, embeddings


def pca_2d(x):
    x_centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    return x_centered @ vt[:2].T


def cosine_neighbors(embeddings, vocab, query, top_k):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normed = embeddings / norms
    word_to_id = {w: i for i, w in enumerate(vocab)}
    if query not in word_to_id:
        return []
    q_idx = word_to_id[query]
    sims = normed @ normed[q_idx]
    best = np.argsort(-sims)
    results = []
    for idx in best:
        if vocab[idx] == query:
            continue
        results.append((vocab[idx], float(sims[idx])))
        if len(results) >= top_k:
            break
    return results


def plot_embeddings(embeddings, vocab, top_words, plot_path):
    coords = pca_2d(embeddings)
    plt.figure(figsize=(9, 7))
    xs = coords[:top_words, 0]
    ys = coords[:top_words, 1]
    plt.scatter(xs, ys, s=40, alpha=0.8)
    for i in range(top_words):
        plt.text(xs[i] + 0.01, ys[i] + 0.01, vocab[i], fontsize=9)
    plt.title("Word Embeddings (PCA to 2D)")
    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, dpi=150)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train or visualize word embeddings.")
    parser.add_argument("--mode", choices=["train", "pretrained"], default="train")
    parser.add_argument("--corpus", type=Path, default=Path("data/toy_corpus.txt"))
    parser.add_argument("--pretrained", type=Path, default=Path("data/pretrained_vectors.txt"))
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--query", type=str, nargs="*", default=[])
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "train":
        vocab, embeddings, _ = train_embeddings(
            args.corpus, args.window, args.dim, args.min_count
        )
    else:
        vocab, embeddings = read_pretrained(args.pretrained)

    top_n = min(args.top_n, len(vocab))
    plot_embeddings(embeddings, vocab, top_n, args.plot)

    for q in args.query:
        neighbors = cosine_neighbors(embeddings, vocab, q, args.neighbors)
        if not neighbors:
            print(f"{q}: not in vocabulary")
            continue
        formatted = ", ".join([f"{w} ({s:.2f})" for w, s in neighbors])
        print(f"{q}: {formatted}")


if __name__ == "__main__":
    main()
