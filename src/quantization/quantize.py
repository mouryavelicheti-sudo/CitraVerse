"""
MODULE V - Evidence-Preserving Edge Optimization: INT8 quantization.

Dynamic quantization (default) works out-of-the-box on any CPU and is the
right first step for edge deployment of a CNN whose bottleneck is the
Linear/Conv layers. Static quantization (calibration-based) is also
provided for lower-latency deployment when representative calibration data
is available.

--------------------------------------------------------------------------
Backend selection (fixed 2026): PyTorch quantized engines are NOT portable
--------------------------------------------------------------------------
`configs/edge.yaml` defaults `quantization.backend` to "qnnpack" because
that's the right engine for actual ARM edge deployment targets. But the
machine RUNNING this quantization step (a training/dev workstation) may be
a completely different PyTorch build that doesn't ship "qnnpack" at all.
In particular: many recent PyTorch CPU wheels for Windows/x86 report

    torch.backends.quantized.supported_engines == ['onednn']

with no "qnnpack" and no "fbgemm" available, so unconditionally setting
`torch.backends.quantized.engine = "qnnpack"` raises
`RuntimeError: quantized engine QNNPACK is not supported` before any
quantization even happens.

`resolve_quantized_backend()` below is the single place that decides which
engine actually gets used: it checks the CURRENT installation's
`torch.backends.quantized.supported_engines` and falls back, in priority
order, to whatever is really available -- never crashing on an
unsupported preference, and never silently claiming the requested backend
ran when a different one actually did. Every caller in this file (and
`scripts/quantize_model.py`) goes through this function instead of setting
`torch.backends.quantized.engine` directly.
"""
import time


# Preference order when the caller's requested backend isn't supported by
# this installation. "qnnpack" and "fbgemm" are the long-standing engines;
# "onednn" is the newer default on several recent PyTorch CPU builds
# (replacing fbgemm as the default in many x86 configurations); "x86" is an
# alias engine on some builds. This list is a fallback preference, not an
# assumption about which one is "best" for edge deployment -- qnnpack is
# still the right target for real ARM edge hardware, this just keeps
# development/training machines that lack it from crashing.
_BACKEND_FALLBACK_PRIORITY = ["qnnpack", "fbgemm", "onednn", "x86"]


def resolve_quantized_backend(preferred: str = "qnnpack", supported=None) -> str:
    """Returns a quantized engine name that is ACTUALLY usable on this
    PyTorch installation, falling back from `preferred` if necessary.

    Args:
        preferred: the backend the caller/config asked for (e.g. from
            configs/edge.yaml -> quantization.backend).
        supported: override for `torch.backends.quantized.supported_engines`,
            used only by tests so the fallback logic can be exercised
            without needing a PyTorch build that actually lacks an engine.
            Production callers should never pass this.

    Raises:
        RuntimeError: if this PyTorch installation reports NO supported
            quantized engines at all (INT8 quantization is not available
            in that environment, full stop -- there is no engine to fall
            back to).
    """
    if supported is None:
        import torch
        supported = list(torch.backends.quantized.supported_engines)

    if not supported:
        raise RuntimeError(
            "This PyTorch installation reports no supported quantized engines "
            "(torch.backends.quantized.supported_engines is empty). INT8 "
            "quantization is not available in this environment."
        )

    if preferred in supported:
        return preferred

    for candidate in _BACKEND_FALLBACK_PRIORITY:
        if candidate in supported:
            return candidate

    # Last resort: something is supported, even if it's not in our priority
    # list above (a future/unknown engine name) -- use it rather than fail.
    return supported[0]


def _set_engine(preferred: str) -> str:
    """Resolves and activates a quantized engine, printing a clear message
    when the actual engine used differs from what was requested (never
    silent about it)."""
    import torch

    resolved = resolve_quantized_backend(preferred)
    if resolved != preferred:
        print(
            f"[src.quantization.quantize] Requested backend '{preferred}' is not supported by "
            f"this PyTorch build (torch.backends.quantized.supported_engines="
            f"{list(torch.backends.quantized.supported_engines)}). Falling back to '{resolved}'."
        )
    torch.backends.quantized.engine = resolved
    return resolved


def dynamic_quantize(model, backend: str = "qnnpack"):
    """Dynamic INT8 quantization. Converts nn.Linear layers only -- this is
    a PyTorch limitation of quantize_dynamic's default module set, not a
    bug here (see scripts/quantize_model.py's layer report, which verifies
    exactly what ended up quantized rather than assuming).

    The quantized module returned has an extra, non-standard attribute
    `quantization_backend_used` set to the ACTUAL engine that was used
    (post-fallback), so callers/tests never have to guess or re-derive it.
    """
    import torch

    resolved_backend = _set_engine(backend)
    model_cpu = model.to("cpu").eval()
    quantized = torch.quantization.quantize_dynamic(
        model_cpu, {torch.nn.Linear}, dtype=torch.qint8
    )
    quantized.quantization_backend_used = resolved_backend
    return quantized


def static_quantize(model, calibration_loader, backend: str = "qnnpack", num_calibration_batches: int = 20):
    """Calibration-based INT8 quantization (Conv2d + Linear, via a fused
    qconfig) -- use this when the research claim needs a genuinely
    mostly-INT8 CNN rather than dynamic quantization's Linear-only
    conversion. See module docstring for backend resolution behavior."""
    import torch

    resolved_backend = _set_engine(backend)
    model_cpu = model.to("cpu").eval()
    model_cpu.qconfig = torch.quantization.get_default_qconfig(resolved_backend)
    torch.quantization.prepare(model_cpu, inplace=True)

    with torch.no_grad():
        for i, (images, _labels, _meta) in enumerate(calibration_loader):
            if i >= num_calibration_batches:
                break
            model_cpu(images)

    torch.quantization.convert(model_cpu, inplace=True)
    model_cpu.quantization_backend_used = resolved_backend
    return model_cpu


def model_size_mb(model) -> float:
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 ** 2)


def benchmark_latency(model, input_tensor, num_warmup: int = 5, num_timed: int = 30) -> dict:
    import torch

    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            model(input_tensor)

        timings = []
        for _ in range(num_timed):
            start = time.perf_counter()
            model(input_tensor)
            timings.append(time.perf_counter() - start)

    import numpy as np
    timings = np.array(timings)
    return {
        "mean_latency_ms": float(timings.mean() * 1000),
        "p50_latency_ms": float(np.percentile(timings, 50) * 1000),
        "p95_latency_ms": float(np.percentile(timings, 95) * 1000),
        "fps": float(1.0 / timings.mean()) if timings.mean() > 0 else None,
    }
