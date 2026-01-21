# Transformer Workshop: Word Embeddings Demo


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

## Train a tiny embedding model

```bash
python src/embedding_demo.py --mode train --corpus data/toy_corpus.txt --window 2 --dim 20 --top-n 30
```

## Train on PTB with a subset + progress

```bash
python src/embedding_demo.py --mode train --corpus ptb.train.txt --max-lines 2000 --progress-every 200
```

```bash
python src/embedding_demo.py --mode train --corpus ptb.train.txt --max-tokens 50000 --progress-every 5000
```

## Use a pretrained embedding file

```bash
python src/embedding_demo.py --mode pretrained --pretrained data/pretrained_vectors.txt --top-n 30
```

## Query nearest neighbors

```bash
python src/embedding_demo.py --mode train --query king queen car
```

## Save a plot instead of showing it

```bash
python src/embedding_demo.py --mode train --plot embeddings.png
```

## Files

- `src/embedding_demo.py`: Training + visualization script.
- `data/toy_corpus.txt`: Small corpus for training in a workshop setting.
- `data/pretrained_vectors.txt`: Tiny pretrained vectors (word2vec text format).
