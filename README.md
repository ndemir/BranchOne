# BranchOne

Python package and scripts for text/image decision-model training and inference.
Use Linux, Python 3.13 and an NVIDIA CUDA GPU. Run these commands from the repository root.

```bash
python scripts/setup_colab.py \
  --venv ../branchone-env \
  --kernel-cache ../branchone-kernels \
  --environment-record ../branchone-work/environment.json
```

Supply your own training and development JSONL files. Each line contains a state,
a question, candidate descriptions and the zero-based correct candidate index:

```json
{"id":"example-1","type":"choice","state":"The item is red.","question":"Which color matches?","candidates":[{"id":"red","description":"Red"},{"id":"blue","description":"Blue"}],"gold_index":0,"group_id":"item-1"}
```

For an image example, add an `image` path relative to `--image-root`.
Keep related examples in the same split and give them the same `group_id`.

```bash
../branchone-env/bin/branchone prepare \
  --split train=../data/train.jsonl \
  --split development=../data/development.jsonl \
  --image-root ../data --output ../prepared

../branchone-env/bin/branchone train \
  --train ../prepared/train.jsonl.gz \
  --development ../prepared/development-panel.jsonl.gz \
  --image-root ../data --config configs/one_epoch.json \
  --output ../runs/fresh
```

To train from existing weights, add `--init-checkpoint ../checkpoint` and use a new
output directory. To resume an interrupted stage, repeat its command with
`--resume`, the same input files and the same output directory.

```bash
../branchone-env/bin/branchone evaluate --help
../branchone-env/bin/branchone predict \
  --checkpoint ../checkpoint --input ../request.json
```

An inference request contains `state`, `question`, `candidates`, and optionally
`image`; omit the training label. Keep data, weights, credentials and generated
outputs outside the repository. Download run artifacts and stop your GPU runtime
when finished.
