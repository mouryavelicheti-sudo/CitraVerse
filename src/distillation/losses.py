"""
MODULE V - Evidence-Preserving Knowledge Distillation losses.

L_student = lambda1 * L_CE + lambda2 * L_KD + lambda3 * L_signature

  L_CE:        standard cross-entropy against ground-truth labels
  L_KD:        L_KD = T^2 * D_KL( P_T^(T) || P_S^(T) ), temperature-softened
               teacher/student probability distributions
  L_signature: L_signature = || Z_T(x) - Z_S(x) ||^2, evidence-signature
               preservation between teacher and student -- see the
               "Z(x) vs. penultimate-feature" note below, this is the
               one term in this file that was previously implemented
               incorrectly and has now been fixed.

--------------------------------------------------------------------------
IMPORTANT (fixed 2026): what Z_T(x) and Z_S(x) actually are
--------------------------------------------------------------------------
Earlier revisions of this file computed L_signature as the MSE between the
teacher's and student's *penultimate CNN embeddings* (whatever the backbone
happened to produce right before its classification head). That is NOT
what the architecture spec means by Z_T(x)/Z_S(x).

Z(x) is defined in Module II (src/evidence/fusion.py) as the FUSED,
multi-component disease evidence signature:

    Z(x) = normalize(concat(Ec(x), Et(x), Es(x), Ed(x)))

This is computed once per image by scripts/extract_evidence.py, entirely
independently of the teacher or student network architecture, and saved to
outputs/features/Z_signatures.npz. Because Z(x) does not depend on which
classifier produced it, the teacher's evidence signature Z_T(x) for a given
image and the "ground-truth" fused Z(x) for that image are the SAME
quantity -- there is nothing model-specific to "extract" from the teacher.

What IS model-specific, and what edge optimization actually threatens, is
whether the STUDENT's internal representation still carries the same
disease-relevant evidence after compression. So the signature-preservation
term in this file is implemented as:

    Z_hat_S(x) = signature_proj(student_penultimate_feat(x))     # [B, Z_dim]
    L_signature = || Z(x) - Z_hat_S(x) ||^2                      # MSE

i.e. Z_T(x) in the loss is the precomputed, model-independent fused
evidence signature (NOT a teacher CNN embedding), and Z_S(x) is a learned
linear projection of the student's own features into that same evidence
space. `signature_proj` is therefore a REQUIRED argument now (previously
optional), since without it there is no way to compare a student feature
vector of arbitrary CNN dimensionality against the fixed-dimensionality
fused evidence vector Z(x).

This distinction matters for the paper: the claim is "the student is
distilled to preserve the project's proposed multi-component disease
evidence signature," not "the student's embeddings are close to the
teacher's embeddings" (which is a much weaker and less interesting claim,
and was previously what the code silently did instead).
"""


def build_distillation_loss(lambda_ce: float, lambda_kd: float, lambda_signature: float, temperature: float):
    total = lambda_ce + lambda_kd + lambda_signature
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"lambda_ce + lambda_kd + lambda_signature must equal 1.0 (got {total}). "
            "Fix configs/training.yaml -> distillation section."
        )

    import torch.nn as nn
    import torch.nn.functional as F

    ce_loss_fn = nn.CrossEntropyLoss()

    def loss_fn(student_logits, teacher_logits, student_feat, z_target, labels, signature_proj):
        """
        student_logits, teacher_logits: [B, num_classes]
        student_feat: [B, D_s] student penultimate features (arbitrary dim,
            backbone-dependent -- e.g. 576 for mobilenet_v3_small)
        z_target: [B, Z_dim] the PRECOMPUTED, model-independent fused
            evidence signature Z(x) for this batch (loaded from
            outputs/features/Z_signatures.npz, aligned by image_path).
            This plays the role of Z_T(x) in the architecture spec.
        labels: [B] ground-truth class indices
        signature_proj: nn.Linear(D_s -> Z_dim). REQUIRED. Maps the
            student's own features into the fused-evidence space so they
            can be compared against z_target on equal footing.
        """
        l_ce = ce_loss_fn(student_logits, labels)

        t = temperature
        p_teacher_t = F.softmax(teacher_logits / t, dim=1)
        log_p_student_t = F.log_softmax(student_logits / t, dim=1)
        l_kd = (t ** 2) * F.kl_div(log_p_student_t, p_teacher_t, reduction="batchmean")

        z_hat_student = signature_proj(student_feat)
        l_signature = F.mse_loss(z_hat_student, z_target.detach())

        total_loss = lambda_ce * l_ce + lambda_kd * l_kd + lambda_signature * l_signature
        return {
            "total": total_loss,
            "L_CE": l_ce.detach().item(),
            "L_KD": l_kd.detach().item(),
            "L_signature": l_signature.detach().item(),
        }

    return loss_fn
