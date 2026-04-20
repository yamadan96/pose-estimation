"""ViTPose model loading utilities."""

import logging

import numpy as np
import torch

# Monkey-patch: transformers ViTPose image processor references `inv` without
# importing it. Inject numpy's inv so scipy_warp_affine doesn't NameError.
import transformers.models.vitpose.image_processing_vitpose as _vp_ip

if not hasattr(_vp_ip, "inv"):
    _vp_ip.inv = np.linalg.inv  # type: ignore[attr-defined]

from transformers import AutoProcessor, VitPoseForPoseEstimation
from transformers.models.vitpose.image_processing_vitpose import VitPoseImageProcessor

logger = logging.getLogger(__name__)

MODEL_ID = "usyd-community/vitpose-base-simple"

# COCO 17 keypoints color palette
PALETTE = [
    [255, 128, 0],
    [255, 153, 51],
    [255, 178, 102],
    [230, 230, 0],
    [255, 153, 255],
    [153, 204, 255],
    [255, 102, 255],
    [255, 51, 255],
    [102, 178, 255],
    [51, 153, 255],
    [255, 153, 153],
    [255, 102, 102],
    [255, 51, 51],
    [153, 255, 153],
    [102, 255, 102],
    [51, 255, 51],
    [0, 255, 0],
    [0, 0, 255],
    [255, 0, 0],
    [255, 255, 255],
]

# Color indices for limb connections (maps to PALETTE)
LINK_COLOR_INDICES = [0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]

# Color indices for keypoints (maps to PALETTE)
KEYPOINT_COLOR_INDICES = [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]

KEYPOINT_SCORE_THRESHOLD = 0.3
KEYPOINT_RADIUS = 4
LIMB_THICKNESS = 2


def load_model(
    device: str = "cuda",
) -> tuple[VitPoseForPoseEstimation, VitPoseImageProcessor]:
    """Load ViTPose model and processor from HuggingFace.

    Args:
        device: Device to load model on ("cuda" or "cpu").

    Returns:
        Tuple of (model, processor).
    """
    resolved_device = device if torch.cuda.is_available() else "cpu"
    if resolved_device != device:
        logger.warning(
            "CUDA not available, falling back to CPU (requested: %s)", device
        )

    logger.info("Loading ViTPose model from %s on %s", MODEL_ID, resolved_device)

    processor: VitPoseImageProcessor = AutoProcessor.from_pretrained(MODEL_ID)
    model: VitPoseForPoseEstimation = VitPoseForPoseEstimation.from_pretrained(MODEL_ID)
    model = model.to(resolved_device)
    model.eval()

    logger.info("Model loaded successfully")
    return model, processor
