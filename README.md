# Cross-Modal Alignment for Zero-Shot Image Classification

Classifies images into categories the model has **never seen during training** by aligning
visual features with text-attribute embeddings in a shared semantic space, instead of relying
on labeled examples for every class.

> Final-year B.Tech project (SRM Institute of Science and Technology, Ramapuram) — reimplemented
> here as a complete, working codebase: transfer-learning CNN backbone, attribute-guided
> cross-attention, cosine-similarity alignment, a training/evaluation pipeline with the standard
> ZSL/GZSL protocol, and a web demo.

## How it works

```
image ──► ResNet backbone ──► spatial feature map (B, C, H, W)
                                        │
                    class attribute vector (e.g. "has red crown", "small beak")
                                        │
                              semantic encoder (MLP)
                                        │
                    attribute-guided cross-attention over the feature map
                    (the text attribute is the query; it "looks at" the image
                     and produces a response vector for that class)
                                        │
              cosine similarity(response, class semantic embedding) ──► score
```

Because classification only needs a class's **attribute vector** (never its images), a class that
had zero training images can still be scored and predicted at test time — that's the zero-shot
part. The same mechanism is what the source report calls an "attention module to get response
maps through feature maps activated by the query of text attribute," paired with a "cosine
distance metric ... to measure the matching degree."

**Training loss** = temperature-scaled cross-entropy over cosine scores (seen classes only) +
a semantic-consistency term pulling each image's response toward its own ground-truth class
embedding — the "novel loss function that encourages semantic consistency between textual
descriptions and synthesized features" from the report's abstract.

## Project layout

```
cross-modal-zsl/
├── src/
│   ├── datasets/
│   │   ├── cub_zsl.py        # CUB-200-2011 loader, attributes, seen/unseen split
│   │   ├── synthetic.py      # offline synthetic dataset for smoke tests / CI
│   │   └── transforms.py     # augmentation: flip, rotate, crop, translate, noise
│   ├── models/
│   │   ├── backbone.py       # transfer-learning ResNet feature extractor
│   │   ├── attention.py      # attribute-guided cross-attention (the "response map")
│   │   ├── semantic_encoder.py
│   │   └── cross_modal_net.py
│   ├── build.py              # wires datasets + loaders together
│   ├── engine.py             # train / eval loops, ZSL + GZSL protocol
│   ├── losses.py
│   └── metrics.py            # per-class accuracy, precision/recall/F1, confusion matrix
├── app/                      # Flask web demo (upload → prediction + attention heatmap)
├── scripts/
│   ├── download_cub.py       # fetches CUB-200-2011
│   └── export_split.py       # documents the seen/unseen class split used for a run
├── configs/default.yaml
├── train.py / evaluate.py / predict.py
└── tests/test_pipeline.py    # end-to-end smoke test, no dataset download required
```

## Setup

```bash
git clone <this-repo-url>
cd cross-modal-zsl
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quickstart (no dataset download, ~1 minute on CPU)

Verify everything works end to end using a small procedurally-generated dataset:

```bash
python train.py --synthetic --epochs 5
python evaluate.py --checkpoint checkpoints/cross_modal_zsl.pt --synthetic
```

or run the automated smoke test:

```bash
pip install -r requirements-dev.txt
pytest tests/
```

The repo also bundles 16 real CUB photographs (`sample_data/`, ~400KB) so you can run the exact
same code path on genuine images with zero download:

```bash
python train.py --data-root sample_data/CUB_sample --epochs 3 --backbone resnet18 --image-size 128
```

See [`sample_data/README.md`](sample_data/README.md) for what's real (the photos) versus
placeholder (the attribute vectors) in that bundle — it's for pipeline verification, not
benchmark numbers.

## Real training on CUB-200-2011

The project targets [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158) (200 bird
species, 312 binary/continuous attributes per class) — the same benchmark referenced in the
report's literature review and existing-system baseline.

```bash
python scripts/download_cub.py --dest data          # downloads + extracts to data/CUB_200_2011
python train.py --data-root data/CUB_200_2011 --epochs 30
python evaluate.py --checkpoint checkpoints/cross_modal_zsl.pt --data-root data/CUB_200_2011
```

By default, 25% of the 200 classes are held out as **unseen** using a fixed random seed (see
`configs/default.yaml: data.unseen_ratio`). To use a literature-standard split instead, provide a
directory with `trainvalclasses.txt` / `testclasses.txt` via `--split-dir`. Run
`python scripts/export_split.py --data-root data/CUB_200_2011 --out splits/default` to save out
whichever split a run actually used, so it's documented and reproducible.

## Evaluation protocol

Following standard zero-shot learning practice, accuracy is **mean per-class accuracy**, not
plain top-1 (test classes are rarely balanced):

- **ZSL** — classify unseen-class images among unseen classes only.
- **GZSL** — classify seen- and unseen-class images among the *union* of both class sets,
  reporting seen accuracy, unseen accuracy, and their harmonic mean **H** (the standard way to
  penalize a model that just always predicts a seen class).

`evaluate.py` also writes macro precision/recall/F1 and a confusion-matrix heatmap to `outputs/`.

## Inference & demo

Single-image CLI:

```bash
python predict.py --checkpoint checkpoints/cross_modal_zsl.pt \
    --data-root data/CUB_200_2011 --image path/to/bird.jpg --candidates unseen
```

Web demo (upload a photo, see top-5 predictions and the attribute-attention heatmap):

```bash
python app/app.py --checkpoint checkpoints/cross_modal_zsl.pt --data-root data/CUB_200_2011
# open http://127.0.0.1:5000
```

## Configuration

All hyperparameters live in `configs/default.yaml` (backbone choice, embedding dimension,
learning rates, loss weights, augmentation, etc.) and can be overridden with CLI flags, e.g.
`--epochs`, `--batch-size`, `--data-root`, `--split-dir`.

## Notes on scope

Attribute-based zero-shot learning needs class-level *semantic* metadata (attributes or text
descriptions), not just images — CUB-200-2011 ships exactly that. To point this at a different
dataset, provide an equivalent `(num_classes, num_attributes)` matrix and reuse
`src/models/semantic_encoder.py` / `src/models/cross_modal_net.py` as-is.

## References

Based on the cross-modal, attention + cosine-similarity ZSL formulation surveyed in the source
report, notably Wu et al., *"A Cross-Modal Alignment for Zero-Shot Image Classification,"* IEEE
Access, 2023, and the standard ZSL/GZSL evaluation protocol of Xian et al., *"Zero-Shot
Learning — A Comprehensive Evaluation of the Good, the Bad and the Ugly,"* IEEE TPAMI, 2019.

## License

MIT — see [LICENSE](LICENSE).
