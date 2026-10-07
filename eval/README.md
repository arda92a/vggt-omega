# Evaluation

Evaluation code for VGGT-Omega on Sintel. This codebase is an agent-assisted
cleanup of our original evaluation implementation. If you encounter any problems, please open a
[GitHub issue](https://github.com/facebookresearch/vggt-omega/issues).

Exact results can vary across machines and
software environments, for example, AUC may differ by up to two points, while the relative
trends between methods remain consistent.

## Install

```bash
pip install -r requirements.txt
pip install -e .
pip install -r eval/requirements.txt
```

## Prepare data

```bash
python eval/prepare.py sintel --output data/sintel
```

Preparation uses local archives when available, otherwise tries the official dataset
server and then a pinned Hugging Face mirror. Use `--source official`, `--source hf`, or
`--source local --archive-dir /path/to/archives` to choose explicitly. To convert an
already extracted official dataset, pass `--raw-root`.

Prepared data has the layout:

```text
data/<dataset>/<scene>/
├── frames.txt
├── cameras.npz
├── images/
└── depths/
```

## Run

```bash
python eval/evaluate.py \
  --dataset sintel \
  --data-root data/sintel \
  --checkpoint /path/to/vggt_omega_1b_416_reproduce.pt \
  --output outputs/sintel.json
```

## Output

The output JSON contains `AUC@3`, `AUC@30`, `delta125`, and `AbsRel`, together with
per-scene values, frame names, dataset counts, and runtime versions. Use
`--max-scenes 1` for a quick smoke run.
