"""MODULE II - Deep Semantic Evidence Ed(x).

Extracts penultimate-layer embeddings from a CNN backbone (default:
ResNet18, ImageNet-pretrained) as the deep semantic evidence component.
Chosen backbones (ResNet18 / MobileNetV3 / EfficientNet-B0) are all viable
edge-deployment candidates, consistent with Module V's requirement that the
architecture used for evidence extraction be compatible with later
distillation/quantization.

Requires torch + torchvision. If unavailable, raises a clear, actionable
ImportError rather than silently degrading to a fake feature (Module II
explicitly forbids faking evidence).
"""
import numpy as np


class DeepEvidenceExtractor:
    def __init__(self, backbone: str = "resnet18", device: str = "cpu", pretrained: bool = True):
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
        except ImportError as e:
            raise ImportError(
                "DeepEvidenceExtractor requires torch and torchvision. "
                "Install with: pip install torch torchvision "
                "(see README.md for platform-specific instructions)."
            ) from e

        self.torch = torch
        self.device = torch.device(device)
        self.backbone_name = backbone

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
            self.feature_dim = net.fc.in_features
            net.fc = nn.Identity()
        elif backbone == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.mobilenet_v3_small(weights=weights)
            self.feature_dim = net.classifier[0].in_features
            net.classifier = nn.Identity()
        elif backbone == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.mobilenet_v3_large(weights=weights)
            self.feature_dim = net.classifier[0].in_features
            net.classifier = nn.Identity()
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            self.feature_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported deep evidence backbone: {backbone}")

        self.net = net.to(self.device).eval()

    def extract_batch(self, image_tensor_batch) -> np.ndarray:
        """image_tensor_batch: torch.FloatTensor [B,3,H,W], ImageNet-normalized.
        Returns numpy array [B, feature_dim]."""
        torch = self.torch
        with torch.no_grad():
            feats = self.net(image_tensor_batch.to(self.device))
        return feats.cpu().numpy()

    def extract_single(self, image_tensor) -> np.ndarray:
        """image_tensor: torch.FloatTensor [3,H,W]."""
        return self.extract_batch(image_tensor.unsqueeze(0))[0]
