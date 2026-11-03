"""
EXPLAINABILITY - Grad-CAM.

Generic Grad-CAM implementation that works with any torchvision-style CNN
backbone by hooking the last convolutional feature map. Supports comparing
explanations before vs. after edge optimization (Module V requirement),
since it accepts any (possibly quantized) model as long as gradients can
flow (quantized INT8 models are inference-only / non-differentiable -- for
those, generate Grad-CAM using the FP32 student and note the INT8 model's
predictions/confidence alongside it; see scripts/generate_gradcam.py).

IMPORTANT: a Grad-CAM heatmap shows where the model's activations
contributed most to a prediction. It does NOT prove causal reasoning about
disease biology -- treat it as a diagnostic aid for the model, not a
scientific claim about the disease itself.
"""
import cv2
import numpy as np


LAST_CONV_LAYER = {
    "resnet18": "layer4",
    "mobilenet_v3_small": "features",   # last block within `features`
    "mobilenet_v3_large": "features",
    "efficientnet_b0": "features",
}


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        import torch

        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # global-average-pooled gradients
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx


def resolve_target_layer(model, backbone_name: str):
    """Returns the actual last-conv-layer module for a torchvision model,
    for the common backbones used in this project."""
    if backbone_name == "resnet18":
        return model.layer4[-1]
    if backbone_name in ("mobilenet_v3_small", "mobilenet_v3_large"):
        return model.features[-1]
    if backbone_name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target layer mapping for backbone: {backbone_name}")


def overlay_heatmap(image_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay
