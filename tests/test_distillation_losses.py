import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.distillation.losses import build_distillation_loss  # noqa: E402


def test_lambdas_must_sum_to_one():
    with pytest.raises(ValueError):
        build_distillation_loss(lambda_ce=0.5, lambda_kd=0.5, lambda_signature=0.5, temperature=4.0)


def test_signature_loss_uses_projected_student_feat_against_z_target_not_teacher_feat():
    """Regression test for the Module V research issue: L_signature must
    compare the student's PROJECTED features against the precomputed fused
    Z(x) target -- it must NOT silently fall back to comparing raw
    teacher/student CNN embeddings against each other.
    """
    torch.manual_seed(0)
    batch, num_classes, feat_dim, z_dim = 4, 3, 8, 5

    loss_fn = build_distillation_loss(lambda_ce=0.4, lambda_kd=0.4, lambda_signature=0.2, temperature=4.0)

    student_logits = torch.randn(batch, num_classes, requires_grad=True)
    teacher_logits = torch.randn(batch, num_classes)
    student_feat = torch.randn(batch, feat_dim, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch,))

    signature_proj = torch.nn.Linear(feat_dim, z_dim)

    # Case 1: z_target exactly matches what the (untrained) projection already
    # outputs -> L_signature should be ~0.
    with torch.no_grad():
        z_target_matching = signature_proj(student_feat)
    result_matching = loss_fn(student_logits, teacher_logits, student_feat, z_target_matching, labels, signature_proj)
    assert result_matching["L_signature"] < 1e-6

    # Case 2: an arbitrary, unrelated z_target of the SAME dimensionality as
    # Z(x) (not the student's own feature dimensionality) -> loss must be
    # computable and nonzero. This is only possible because L_signature is
    # defined against a Z_dim-shaped target via signature_proj, not against
    # feat_dim-shaped teacher features directly.
    z_target_random = torch.randn(batch, z_dim)
    result_random = loss_fn(student_logits, teacher_logits, student_feat, z_target_random, labels, signature_proj)
    assert result_random["L_signature"] > 0.0


def test_total_loss_is_finite_and_differentiable():
    torch.manual_seed(1)
    batch, num_classes, feat_dim, z_dim = 4, 3, 8, 5
    loss_fn = build_distillation_loss(lambda_ce=0.4, lambda_kd=0.4, lambda_signature=0.2, temperature=4.0)

    student_logits = torch.randn(batch, num_classes, requires_grad=True)
    teacher_logits = torch.randn(batch, num_classes)
    student_feat = torch.randn(batch, feat_dim, requires_grad=True)
    z_target = torch.randn(batch, z_dim)
    labels = torch.randint(0, num_classes, (batch,))
    signature_proj = torch.nn.Linear(feat_dim, z_dim)

    result = loss_fn(student_logits, teacher_logits, student_feat, z_target, labels, signature_proj)
    assert torch.isfinite(result["total"])
    result["total"].backward()
    assert student_logits.grad is not None
    assert student_feat.grad is not None
