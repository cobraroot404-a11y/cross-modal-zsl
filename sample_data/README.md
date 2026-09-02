# Bundled sample data

16 real photographs (2 species, 8 images each) pulled directly from the official
[CUB-200-2011](https://data.caltech.edu/records/65de6-vp158) archive, resized to a 400px
longest side, so this repo is runnable out of the box without any download.

**What's real:** the photographs themselves and their species identity.

**What's a placeholder:** the attribute vectors in `attributes/class_attribute_labels_continuous.txt`.
Getting the *official* 312-dim CUB attribute annotations for a couple of classes still requires
pulling them out of the full ~1.1GB archive, so this sample instead ships 16-dim randomly
generated placeholder vectors — enough to exercise every part of the pipeline (data loading,
augmentation, the attention/alignment model, training, ZSL/GZSL evaluation) end to end, but
**not meaningful for reporting accuracy numbers**.

Only 2 classes means every ZSL/GZSL metric on this sample is trivial (1 class per split) — it's
a smoke test, not a benchmark. For real, reportable results, run:

```bash
python scripts/download_cub.py --dest data
python train.py --data-root data/CUB_200_2011 --epochs 30
```

Try the bundled sample with:

```bash
python train.py --data-root sample_data/CUB_sample --epochs 3 --backbone resnet18 --image-size 128
```
