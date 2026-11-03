import pytest

torch = pytest.importorskip("torch")

from src.quantization.quantize import (  # noqa: E402
    benchmark_latency,
    dynamic_quantize,
    model_size_mb,
    resolve_quantized_backend,
)


class _TinyNet(torch.nn.Module):
    """Deliberately tiny (~50 parameters). Used to test quantization
    CORRECTNESS (does Linear actually convert, does Conv correctly stay
    FP32, does the forward pass still run) -- NOT model-size reduction.
    At this scale, `torch.quantization.quantize_dynamic`'s per-tensor
    quantization metadata (scale, zero-point, packed-params wrapper
    objects) is fixed overhead that is NOT proportional to tensor size, so
    it can easily outweigh the few bytes actually saved by storing a
    handful of weights as int8 instead of fp32. That is expected, correct
    behavior for a toy model, not a quantization defect -- see
    test_dynamic_quantize_reduces_size_on_a_parameter_heavy_model for
    where size reduction IS a meaningful, testable claim.
    """
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class _ParameterHeavyNet(torch.nn.Module):
    """A model whose Linear layers hold enough parameters (~8.4M) that
    fp32->int8 weight compression (4 bytes/weight -> 1 byte/weight, a real,
    ~4x effect) clearly dominates any fixed per-tensor quantization
    metadata overhead. This is the model actual model-size-reduction
    claims should be tested against -- see
    test_dynamic_quantize_reduces_size_on_a_parameter_heavy_model.
    """
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Linear(4, 2048)
        self.fc2 = torch.nn.Linear(2048, 2048)   # ~4.2M params, the dominant term
        self.fc3 = torch.nn.Linear(2048, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


@pytest.fixture(scope="module")
def backend():
    """The quantized engine actually usable on THIS machine's PyTorch
    build. Never hardcode "qnnpack" here: `torch.backends.quantized.
    supported_engines` varies by platform/build -- e.g. many Windows/x86
    CPU wheels report only ['onednn'] and don't ship "qnnpack" at all, so
    a hardcoded "qnnpack" would fail with
    `RuntimeError: quantized engine QNNPACK is not supported` on those
    machines even though quantization itself works fine there.
    """
    return resolve_quantized_backend("qnnpack")


# --- Backend resolution itself (the actual bug fix) --------------------

def test_resolve_quantized_backend_returns_a_currently_supported_engine():
    """Whatever this function returns for the REAL installation must
    actually be usable -- i.e. must be a member of
    torch.backends.quantized.supported_engines on this machine."""
    resolved = resolve_quantized_backend("qnnpack")
    assert resolved in torch.backends.quantized.supported_engines


def test_resolve_quantized_backend_prefers_the_requested_backend_when_available():
    # If "qnnpack" happens to be supported here, it should be returned
    # unchanged rather than overridden by the fallback priority list.
    if "qnnpack" in torch.backends.quantized.supported_engines:
        assert resolve_quantized_backend("qnnpack") == "qnnpack"
    else:
        pytest.skip("qnnpack is not available on this machine -- covered by the fallback test instead.")


def test_resolve_quantized_backend_falls_back_when_requested_backend_unavailable():
    """Simulates exactly this bug report's environment (supported_engines
    == ['onednn'], no qnnpack/fbgemm) via the `supported=` override, so the
    fallback logic is tested deterministically regardless of what THIS
    machine's real PyTorch build happens to support.
    """
    resolved = resolve_quantized_backend("qnnpack", supported=["onednn"])
    assert resolved == "onednn"


def test_resolve_quantized_backend_respects_fallback_priority_order():
    # fbgemm should be preferred over onednn/x86 when qnnpack is unavailable
    # but fbgemm is, per the documented fallback priority.
    resolved = resolve_quantized_backend("qnnpack", supported=["x86", "onednn", "fbgemm"])
    assert resolved == "fbgemm"


def test_resolve_quantized_backend_raises_when_nothing_is_supported():
    with pytest.raises(RuntimeError):
        resolve_quantized_backend("qnnpack", supported=[])


# --- Quantization itself, using whatever backend this machine supports --

def test_dynamic_quantize_converts_linear_but_may_leave_conv_fp32(backend):
    model = _TinyNet().eval()
    quantized = dynamic_quantize(model, backend=backend)

    fc_module = quantized.fc
    assert "quantized" in fc_module.__class__.__module__, (
        "Dynamic quantization should convert the Linear layer to a quantized module."
    )
    # This assertion documents the caveat scripts/quantize_model.py reports:
    # dynamic quantization does NOT touch Conv2d, regardless of which
    # backend engine (qnnpack/fbgemm/onednn/x86) is used to run it.
    conv_module = quantized.conv
    assert "quantized" not in conv_module.__class__.__module__

    # The actually-used backend is stamped onto the returned module so
    # callers/tests never have to guess (or re-derive) which engine ran.
    assert quantized.quantization_backend_used == backend


def test_quantized_tiny_model_runs_forward_correctly(backend):
    """The tiny model's role in this suite is verifying CORRECTNESS of
    quantization -- the forward pass still runs and produces the right
    output shape/values on real (finite) numbers -- not size reduction.

    A model-size assertion does not belong on `_TinyNet`: with ~50 total
    parameters, `quantize_dynamic`'s fixed per-tensor quantization
    metadata (scale, zero-point, and the packed-params wrapper it
    allocates) is not proportional to tensor size, so it can easily
    outweigh the handful of bytes actually saved by storing 8 float32
    weights as int8 instead. Asserting "quantized <= original size" as a
    universal invariant is therefore not scientifically defensible at this
    scale -- it conflates fixed bookkeeping overhead with the actual
    compression effect being tested. See
    test_dynamic_quantize_reduces_size_on_a_parameter_heavy_model for
    where that claim IS tested, on a model where it's a meaningful effect
    rather than noise.
    """
    model = _TinyNet().eval()
    quantized = dynamic_quantize(model, backend=backend)

    dummy = torch.randn(1, 3, 16, 16)
    out = quantized(dummy)
    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()


def test_dynamic_quantize_reduces_size_on_a_parameter_heavy_model(backend):
    """Model-size reduction from dynamic INT8 quantization is a real,
    measurable effect (int8 weight storage is 1/4 the bytes of fp32), but
    it is only a meaningful claim once Linear-layer weight storage
    dominates the model's total serialized size -- which `_ParameterHeavyNet`
    (~8.4M Linear parameters) is specifically built to guarantee, unlike
    `_TinyNet` above.
    """
    model = _ParameterHeavyNet().eval()
    fp32_size = model_size_mb(model)
    quantized = dynamic_quantize(model, backend=backend)
    int8_size = model_size_mb(quantized)

    assert int8_size < fp32_size, (
        f"Expected INT8 size ({int8_size:.4f} MB) to be smaller than FP32 size "
        f"({fp32_size:.4f} MB) on a parameter-heavy model, where Linear-layer "
        f"weight storage dominates any fixed quantization metadata overhead."
    )
    # A model dominated by Linear weights should approach the theoretical
    # ~4x reduction (fp32: 4 bytes/weight -> int8: 1 byte/weight). Assert a
    # conservative bound (>2x) rather than the exact ratio, since biases
    # stay fp32 and quantization metadata adds a small, fixed amount back.
    assert int8_size < fp32_size * 0.5, (
        f"Expected at least a ~2x size reduction on a Linear-dominated model "
        f"(got FP32={fp32_size:.4f} MB -> INT8={int8_size:.4f} MB), which would "
        f"suggest the Linear layers were not actually converted to int8."
    )

    dummy = torch.randn(1, 3, 16, 16)
    out = quantized(dummy)
    assert out.shape == (1, 10)
    assert torch.isfinite(out).all()


def test_benchmark_latency_reports_expected_keys():
    model = _TinyNet().eval()
    dummy = torch.randn(1, 3, 16, 16)
    result = benchmark_latency(model, dummy, num_warmup=1, num_timed=3)
    for key in ("mean_latency_ms", "p50_latency_ms", "p95_latency_ms", "fps"):
        assert key in result
    assert result["mean_latency_ms"] >= 0


# --- Regression tests: DiseaseClassifier feature hook vs. quantization -----
#
# Bug: `python run_pipeline.py --quantize` failed with
#     Can't get local object '_get_feature_extractor.<locals>.hook'
# Root cause: DiseaseClassifier.build() attaches a forward hook to the
# backbone whose callback is a LOCAL closure (`_get_feature_extractor.
# <locals>.hook` in src/models/classifier.py), used only for penultimate-
# feature extraction during classification/distillation. Both
# dynamic_quantize() and static_quantize() internally deep-copy the model
# before converting it, which carries the hook (and its unpicklable local
# function) onto the returned quantized module. scripts/quantize_model.py's
# later `torch.save(quantized_net, int8_path)` then fails trying to pickle
# that local function. Fix: scripts/quantize_model.py now removes
# `student_fp32._hook_handle` right after loading the checkpoint, before
# quantization ever runs.

def _tiny_disease_classifier():
    """A real DiseaseClassifier (not a fake), on the smallest supported
    backbone, with pretrained=False so no network access is needed."""
    from src.models.classifier import DiseaseClassifier

    pytest.importorskip("torchvision")
    return DiseaseClassifier.build("resnet18", num_classes=3, pretrained=False)


def test_A_diseaseclassifier_feature_hook_can_be_removed_before_quantization():
    """A. A DiseaseClassifier-created backbone's feature hook can be located
    and removed cleanly before quantization."""
    clf = _tiny_disease_classifier()

    # The hook is genuinely registered on the backbone submodule that feeds
    # the classifier head (avgpool for resnet18) before we do anything.
    assert clf._hook_handle is not None
    assert len(clf.net.avgpool._forward_hooks) == 1

    clf._hook_handle.remove()

    # After removal, the backbone module has no forward hooks left, and the
    # handle is idempotent/safe to remove twice (mirrors the `getattr(...,
    # None) is not None` guard used in scripts/quantize_model.py -- calling
    # .remove() again must not raise).
    assert len(clf.net.avgpool._forward_hooks) == 0
    clf._hook_handle.remove()  # no-op, must not raise

    # The plain backbone (what quantization actually operates on) still runs.
    clf.net.eval()
    dummy = torch.randn(1, 3, 32, 32)
    out = clf.net(dummy)
    assert out.shape == (1, 3)
    assert torch.isfinite(out).all()


def test_B_quantized_model_with_hook_removed_saves_with_torch_save(tmp_path, backend):
    """B. After removing the feature hook (as scripts/quantize_model.py now
    does), a dynamically quantized DiseaseClassifier backbone can be
    torch.save()'d without the local-hook pickle error -- this is the exact
    failure mode from `python run_pipeline.py --quantize`."""
    clf = _tiny_disease_classifier()
    clf.eval()
    clf._hook_handle.remove()

    quantized = dynamic_quantize(clf.net, backend=backend)

    save_path = tmp_path / "student_int8.pt"
    torch.save(quantized, save_path)  # must NOT raise
    assert save_path.exists() and save_path.stat().st_size > 0

    # And it round-trips: the saved quantized module can be loaded back and
    # still produces a valid forward pass.
    reloaded = torch.load(save_path, weights_only=False)
    dummy = torch.randn(1, 3, 32, 32)
    out = reloaded(dummy)
    assert out.shape == (1, 3)
    assert torch.isfinite(out).all()


def test_B_negative_saving_with_hook_still_attached_reproduces_the_original_bug(tmp_path, backend):
    """Companion to test_B above: confirms the diagnosis. If the feature
    hook is left attached (the pre-fix state), torch.save on the quantized
    module fails to pickle the local hook closure -- proving the hook, not
    something else, was the actual cause of the reported error."""
    clf = _tiny_disease_classifier()
    clf.eval()
    # Intentionally do NOT remove clf._hook_handle here.

    quantized = dynamic_quantize(clf.net, backend=backend)

    save_path = tmp_path / "student_int8_broken.pt"
    with pytest.raises((AttributeError, TypeError, Exception)) as excinfo:
        torch.save(quantized, save_path)
    # The specific reported message mentions the local hook object; accept
    # any pickling failure that references it, so this doesn't overfit to
    # one exact exception class across Python/torch versions.
    assert "hook" in str(excinfo.value) or "local object" in str(excinfo.value) or "pickle" in str(excinfo.value).lower()


def test_C_static_quantization_calibration_dataloader_uses_agriverse_collate():
    """C. scripts/quantize_model.py's static-quantization calibration
    DataLoader must use collate_fn=agriverse_collate, so switching
    configs/edge.yaml from dynamic to static later does not resurrect the
    mixed string/NaN metadata collate crash fixed in
    tests/test_image_dataset.py."""
    import os

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(repo_root, "scripts", "quantize_model.py"), "r", encoding="utf-8") as f:
        source = f.read()

    assert "from src.data.image_dataset import AgriVerseImageDataset, agriverse_collate, build_transforms" in source

    marker = "calib_loader = DataLoader("
    open_paren_idx = source.index(marker) + len(marker) - 1
    depth = 0
    end = None
    for i in range(open_paren_idx, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    assert end is not None, "Could not find closing paren for calib_loader = DataLoader(...)"
    call_text = source[open_paren_idx:end]
    assert "collate_fn=agriverse_collate" in call_text


def test_D_existing_dynamic_quantize_behavior_is_unchanged(backend):
    """D. Sanity check that fixing the hook/pickle bug did not touch
    dynamic_quantize's actual conversion behavior: Linear -> quantized,
    Conv2d stays FP32, exactly as before this change (mirrors
    test_dynamic_quantize_converts_linear_but_may_leave_conv_fp32 above,
    kept here as an explicit "nothing regressed" checkpoint for this fix)."""
    model = _TinyNet().eval()
    quantized = dynamic_quantize(model, backend=backend)
    assert "quantized" in quantized.fc.__class__.__module__
    assert "quantized" not in quantized.conv.__class__.__module__
    assert quantized.quantization_backend_used == backend
