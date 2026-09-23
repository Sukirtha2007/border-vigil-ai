"""
Night Processor — auto-detects dark frames and applies image enhancement
to improve detection accuracy in low-light / night-time conditions.

Two enhancement strategies:
  1. CLAHE (Contrast Limited Adaptive Histogram Equalization) — boosts contrast
     in dark regions without blowing out bright areas.
  2. Brightness boost — simple gamma correction for extremely dark frames.

Usage:
    night = NightProcessor()
    enhanced_frame = night.process(frame)  # returns enhanced frame if dark, else original
"""

import cv2
import numpy as np


class NightProcessor:
    def __init__(self,
                 darkness_threshold: float = 60.0,
                 very_dark_threshold: float = 30.0,
                 clahe_clip_limit: float = 3.0,
                 clahe_grid_size: tuple = (8, 8),
                 gamma_boost: float = 2.0,
                 mode: str = "auto"):
        """
        Args:
            darkness_threshold: mean brightness below which CLAHE is applied
            very_dark_threshold: mean brightness below which additional gamma boost is applied
            clahe_clip_limit: CLAHE clip limit (higher = more contrast)
            clahe_grid_size: CLAHE tile grid size
            gamma_boost: gamma value for very dark frames (>1 = brighter)
            mode: "auto" (detect darkness), "on" (always enhance), "off" (never enhance)
        """
        self.darkness_threshold = darkness_threshold
        self.very_dark_threshold = very_dark_threshold
        self.gamma_boost = gamma_boost
        self.mode = mode

        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_grid_size
        )

        # Pre-compute gamma lookup table
        self._gamma_lut = self._build_gamma_lut(gamma_boost)

        # Tracking
        self.is_night_mode = False
        self.current_brightness = 0.0

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        """Build lookup table for gamma correction."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in range(256)]).astype("uint8")
        return table

    def _compute_brightness(self, frame: np.ndarray) -> float:
        """Mean brightness of the frame (grayscale)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE on the L channel of LAB color space."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L (lightness) channel
        l_enhanced = self._clahe.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def _apply_gamma(self, frame: np.ndarray) -> np.ndarray:
        """Apply gamma correction for very dark frames."""
        return cv2.LUT(frame, self._gamma_lut)

    def _apply_denoising(self, frame: np.ndarray) -> np.ndarray:
        """Light denoising — night frames tend to be noisy after enhancement."""
        return cv2.fastNlMeansDenoisingColored(frame, None, 6, 6, 7, 21)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame. If dark enough, apply night-mode enhancement.
        Returns the (possibly enhanced) frame.
        """
        if self.mode == "off":
            self.is_night_mode = False
            return frame

        self.current_brightness = self._compute_brightness(frame)

        if self.mode == "on" or self.current_brightness < self.darkness_threshold:
            self.is_night_mode = True
            enhanced = self._apply_clahe(frame)

            # Extra boost for very dark frames
            if self.current_brightness < self.very_dark_threshold:
                enhanced = self._apply_gamma(enhanced)

            return enhanced
        else:
            self.is_night_mode = False
            return frame
