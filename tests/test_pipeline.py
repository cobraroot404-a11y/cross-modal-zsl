"""End-to-end smoke test using the synthetic dataset -- no download needed.

Exercises: data loading -> augmentation -> forward pass -> loss -> backward
-> optimizer step -> zero-shot evaluation, and checks that the loss actually
decreases, i.e. the cross-modal alignment objective has real learning signal.
"""
import torch

from src.build import build_bundle
from src.engine import evaluate_zero_shot, train_one_epoch
from src.losses import CrossModalAlignmentLoss
from src.models.cross_modal_net import CrossModalZSLNet


def test_synthetic_pipeline_learns():
    torch.manual_seed(0)
    device = torch.device("cpu")

    bundle = build_bundle(
        synthetic=True,
        data_root="unused",
        split_dir=None,
        unseen_ratio=0.3,
        image_size=64,
        batch_size=8,
        eval_batch_size=8,
        num_workers=0,
        seed=0,
        synthetic_num_classes=10,
        synthetic_attr_dim=16,
        synthetic_samples_per_class=8,
    )

    assert len(bundle.seen_class_ids) == 7
    assert len(bundle.unseen_class_ids) == 3

    model = CrossModalZSLNet(
        attribute_dim=bundle.attribute_dim,
        backbone="resnet18",
        pretrained=False,
        freeze_backbone_stages=0,
        embed_dim=64,
        attribute_hidden_dim=64,
    ).to(device)

    criterion = CrossModalAlignmentLoss(temperature=0.1, consistency_weight=0.5, label_smoothing=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    seen_attrs = bundle.attribute_tensor_fn(bundle.seen_class_ids)

    losses = []
    for epoch in range(1, 4):
        stats = train_one_epoch(
            model, bundle.train_loader, seen_attrs, criterion, optimizer, device,
            grad_clip=5.0, log_every=1000, epoch=epoch,
        )
        losses.append(stats["loss"])

    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"

    results = evaluate_zero_shot(
        model, device,
        seen_loader=bundle.test_seen_loader,
        unseen_loader=bundle.test_unseen_loader,
        seen_class_ids=bundle.seen_class_ids,
        unseen_class_ids=bundle.unseen_class_ids,
        attribute_tensor_fn=bundle.attribute_tensor_fn,
    )

    assert 0.0 <= results["zsl"]["accuracy"] <= 1.0
    assert 0.0 <= results["gzsl"]["harmonic_mean"] <= 1.0


if __name__ == "__main__":
    test_synthetic_pipeline_learns()
    print("OK")
