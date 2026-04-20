"""Singleton predictor for ViTPose human pose estimation."""

import logging
import math

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from transformers import VitPoseForPoseEstimation
from transformers.models.vitpose.image_processing_vitpose import VitPoseImageProcessor

from .model import (
    KEYPOINT_COLOR_INDICES,
    KEYPOINT_RADIUS,
    KEYPOINT_SCORE_THRESHOLD,
    LIMB_THICKNESS,
    LINK_COLOR_INDICES,
    PALETTE,
    load_model,
)

logger = logging.getLogger(__name__)


def _draw_keypoints(
    image: NDArray[np.uint8],
    keypoints: NDArray[np.float64],
    scores: NDArray[np.float64],
    keypoint_colors: NDArray[np.int64],
    radius: int,
    threshold: float,
) -> None:
    """Draw keypoint circles on the image.

    Args:
        image: BGR image array (H, W, 3), modified in-place.
        keypoints: Keypoint coordinates of shape (num_keypoints, 2).
        scores: Confidence scores of shape (num_keypoints,).
        keypoint_colors: RGB color array of shape (num_keypoints, 3).
        radius: Circle radius in pixels.
        threshold: Minimum score to draw a keypoint.
    """
    for kid, (kpt, kpt_score) in enumerate(zip(keypoints, scores, strict=True)):
        if kpt_score <= threshold:
            continue
        x_coord, y_coord = int(kpt[0]), int(kpt[1])
        color = tuple(int(c) for c in keypoint_colors[kid])
        cv2.circle(image, (x_coord, y_coord), radius, color, -1)


def _draw_limbs(
    image: NDArray[np.uint8],
    keypoints: NDArray[np.float64],
    scores: NDArray[np.float64],
    edges: list[tuple[int, int]],
    link_colors: NDArray[np.int64],
    thickness: int,
    threshold: float,
) -> None:
    """Draw skeleton limb connections on the image.

    Args:
        image: BGR image array (H, W, 3), modified in-place.
        keypoints: Keypoint coordinates of shape (num_keypoints, 2).
        scores: Confidence scores of shape (num_keypoints,).
        edges: List of (start_idx, end_idx) pairs defining limb connections.
        link_colors: RGB color array of shape (num_edges, 3).
        thickness: Line thickness in pixels.
        threshold: Minimum score for both endpoints to draw a limb.
    """
    height, width, _ = image.shape
    for sk_id, (start_idx, end_idx) in enumerate(edges):
        x1, y1 = int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1])
        x2, y2 = int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1])
        score1 = scores[start_idx]
        score2 = scores[end_idx]

        in_bounds = (
            0 < x1 < width and 0 < y1 < height and 0 < x2 < width and 0 < y2 < height
        )
        above_threshold = score1 > threshold and score2 > threshold

        if not (in_bounds and above_threshold):
            continue

        color = tuple(int(c) for c in link_colors[sk_id])
        mean_x = int(np.mean([x1, x2]))
        mean_y = int(np.mean([y1, y2]))
        length = int(((y1 - y2) ** 2 + (x1 - x2) ** 2) ** 0.5)
        angle = int(math.degrees(math.atan2(y1 - y2, x1 - x2)))
        stick_width = 2
        polygon = cv2.ellipse2Poly(
            (mean_x, mean_y),
            (length // 2, stick_width),
            angle,
            0,
            360,
            1,
        )
        cv2.fillConvexPoly(image, polygon, color)


class Predictor:
    """Singleton predictor for ViTPose pose estimation.

    Usage:
        predictor = Predictor()
        predictor.initialize(device="cuda")
        result_image = predictor.predict(input_image)
    """

    _instance: "Predictor | None" = None
    _initialized: bool = False

    def __new__(cls) -> "Predictor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_model"):
            self._model: VitPoseForPoseEstimation | None = None
            self._processor: VitPoseImageProcessor | None = None
            self._device: str = "cpu"

    def initialize(self, device: str = "cuda") -> None:
        """Load the model and processor.

        Args:
            device: Device string ("cuda" or "cpu").
        """
        if self._initialized:
            logger.info("Predictor already initialized, skipping")
            return

        self._device = device
        self._model, self._processor = load_model(device)
        self._initialized = True
        logger.info("Predictor initialized on %s", self._device)

    @property
    def model(self) -> VitPoseForPoseEstimation:
        """Return the loaded model, raising if not initialized."""
        if self._model is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")
        return self._model

    @property
    def processor(self) -> VitPoseImageProcessor:
        """Return the loaded processor, raising if not initialized."""
        if self._processor is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")
        return self._processor

    def predict(self, image: Image.Image) -> Image.Image:
        """Detect keypoints and draw skeleton overlay on the image.

        Uses the full image as a single bounding box (single-person mode).
        For multi-person detection, an object detector would be needed upstream.

        Args:
            image: Input PIL image.

        Returns:
            PIL image with skeleton overlay drawn.
        """
        if not self._initialized:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")

        image_rgb = image.convert("RGB")
        width, height = image_rgb.size

        # Use the full image as a single bounding box (COCO format: x, y, w, h)
        boxes = [[[0.0, 0.0, float(width), float(height)]]]

        inputs = self.processor(image_rgb, boxes=boxes, return_tensors="pt").to(
            self.model.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        pose_results = self.processor.post_process_pose_estimation(outputs, boxes=boxes)
        image_pose_result = pose_results[0]

        # Get skeleton edges from model config
        edges: list[tuple[int, int]] = self.model.config.edges

        # Prepare color arrays
        palette = np.array(PALETTE)
        link_colors = palette[LINK_COLOR_INDICES]
        keypoint_colors = palette[KEYPOINT_COLOR_INDICES]

        # Draw on a copy of the image (OpenCV uses BGR)
        numpy_image = np.array(image_rgb)
        canvas = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        for pose_result in image_pose_result:
            keypoints = np.array(pose_result["keypoints"])
            scores = np.array(pose_result["scores"])

            _draw_limbs(
                canvas,
                keypoints,
                scores,
                edges,
                link_colors,
                thickness=LIMB_THICKNESS,
                threshold=KEYPOINT_SCORE_THRESHOLD,
            )
            _draw_keypoints(
                canvas,
                keypoints,
                scores,
                keypoint_colors,
                radius=KEYPOINT_RADIUS,
                threshold=KEYPOINT_SCORE_THRESHOLD,
            )

        # Convert back to RGB for PIL
        result_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
