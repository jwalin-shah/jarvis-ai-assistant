"""Feature extraction interface contracts.

Defines a common protocol for feature extractors used across classifiers.
This enables composable, testable feature pipelines without coupling
consumers to specific implementations.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class FeatureExtractor(Protocol):
    """Protocol for feature extraction components.

    All feature extractors (category, relationship, mobilization, etc.)
    can implement this interface to enable composable pipelines.

    The extract() method returns a flat numpy feature vector suitable
    for input to sklearn/lightgbm classifiers.
    """

    @property
    def feature_dim(self) -> int:
        """Total number of features produced by this extractor."""
        ...

    @property
    def feature_names(self) -> list[str]:
        """Human-readable names for each feature dimension."""
        ...

    def extract(
        self,
        text: str,
        context: list[str] | None = None,
        **kwargs: Any,
    ) -> NDArray[np.float32]:
        """Extract features from text.

        Args:
            text: Input text to extract features from.
            context: Optional preceding message context.
            **kwargs: Extractor-specific parameters (e.g., mob_pressure, mob_type).

        Returns:
            1-D numpy array of shape (feature_dim,) with extracted features.
        """
        ...
