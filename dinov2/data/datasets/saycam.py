# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, List, Optional

import numpy as np

from .extended import ExtendedVisionDataset


class SAYCam(ExtendedVisionDataset):
    def __init__(
        self,
        *,
        manifest: str,
        root: str = "",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._manifest_path = manifest
        self._root = root
        self._image_paths = self._load_manifest(manifest, root)

    @staticmethod
    def _load_manifest(manifest_path: str, root: str) -> List[str]:
        if not os.path.isfile(manifest_path):
            raise RuntimeError(f'manifest not found: "{manifest_path}"')

        image_paths: List[str] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if os.path.isabs(line):
                    image_paths.append(line)
                else:
                    image_paths.append(os.path.join(root, line))

        if not image_paths:
            raise RuntimeError(f'manifest has no usable paths: "{manifest_path}"')

        return image_paths

    def get_image_relpath(self, index: int) -> str:
        return self._image_paths[index]

    def get_image_data(self, index: int) -> bytes:
        image_full_path = self.get_image_relpath(index)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Any:
        # SAYCam pretraining is self-supervised; targets are placeholders.
        return 0

    def get_targets(self) -> np.ndarray:
        return np.zeros(len(self._image_paths), dtype=np.int64)

    def __len__(self) -> int:
        return len(self._image_paths)
