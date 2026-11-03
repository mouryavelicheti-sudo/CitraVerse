"""
MODULE IV - Disease classifier.

Wraps a backbone so forward() returns BOTH the class-probability vector
P(y|x) and the penultimate-layer embedding (used as Z_model(x), which is
compared against Z_S(x)/Z_T(x) evidence signatures during distillation and
against Grad-CAM target layers during explainability).
"""


def _get_feature_extractor(net, backbone_name: str):
    """Returns a function(x) -> penultimate features, by temporarily
    swapping the final classification layer for nn.Identity() at call time
    is wasteful; instead we register a forward hook on the layer that feeds
    the final classifier head."""
    import torch.nn as nn

    features = {}

    def hook(module, inp, out):
        features["feat"] = out

    if backbone_name == "resnet18":
        # `avgpool` output flattened is what feeds `fc`.
        handle = net.avgpool.register_forward_hook(hook)
    elif backbone_name in ("mobilenet_v3_small", "mobilenet_v3_large"):
        handle = net.avgpool.register_forward_hook(hook)
    elif backbone_name == "efficientnet_b0":
        handle = net.avgpool.register_forward_hook(hook)
    else:
        raise ValueError(f"No feature hook defined for backbone: {backbone_name}")

    return features, handle


class DiseaseClassifier:
    """Thin functional wrapper (not an nn.Module subclass, to keep this file
    importable/reviewable without requiring torch at import time) around a
    torch backbone. Construct via `DiseaseClassifier.build(...)`.
    """

    def __init__(self, net, backbone_name: str):
        self.net = net
        self.backbone_name = backbone_name
        self._features, self._hook_handle = _get_feature_extractor(net, backbone_name)

    @classmethod
    def build(cls, backbone_name: str, num_classes: int, pretrained: bool = True):
        from src.models.backbones import build_backbone

        net = build_backbone(backbone_name, num_classes, pretrained)
        return cls(net, backbone_name)

    def forward(self, x):
        """Returns (logits, probabilities, penultimate_features_flattened)."""
        import torch

        logits = self.net(x)
        probs = torch.softmax(logits, dim=1)
        feat = self._features["feat"]
        feat_flat = torch.flatten(feat, 1)
        return logits, probs, feat_flat

    def __call__(self, x):
        return self.forward(x)

    def to(self, device):
        self.net = self.net.to(device)
        return self

    def train(self):
        self.net.train()
        return self

    def eval(self):
        self.net.eval()
        return self

    def parameters(self):
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)
