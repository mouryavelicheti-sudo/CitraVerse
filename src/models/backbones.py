"""Backbone factory shared by the teacher (FP32, strongest practical model)
and the student (lightweight, edge-friendly model) in Module V, and reused
as the Deep Semantic Evidence backbone in Module II so evidence and
classification stay consistent.
"""


def build_backbone(name: str, num_classes: int, pretrained: bool = True):
    try:
        import torch.nn as nn
        import torchvision.models as models
    except ImportError as e:
        raise ImportError(
            "build_backbone requires torch + torchvision. "
            "Install with: pip install torch torchvision"
        ) from e

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        feature_dim = 512
    elif name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
        in_features = net.classifier[0].in_features
        net.classifier = nn.Sequential(
            nn.Linear(in_features, 256), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(256, num_classes)
        )
        feature_dim = in_features
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)
        in_features = net.classifier[0].in_features
        net.classifier = nn.Sequential(
            nn.Linear(in_features, 256), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(256, num_classes)
        )
        feature_dim = in_features
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features
        net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_classes))
        feature_dim = in_features
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    net.evidence_feature_dim = feature_dim
    return net
