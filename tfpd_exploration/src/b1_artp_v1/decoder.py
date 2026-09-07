"""Official-shape B1 decoder for a prebuilt six-date ARTP payload map."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.interface import BCIDecoder

from tfpd_exploration.src.b1_sfcj_v1.decoder import assert_predict_shape, resolve_dataset_tag

from .core import ARTPConfig, predict_from_payload


class B1ARTPDecoder(BCIDecoder):
    def __init__(
        self,
        payloads: dict,
        *,
        config: ARTPConfig = ARTPConfig(),
        task_config: Optional[FalconConfig] = None,
        batch_size: int = 1,
    ):
        super().__init__(task_config or FalconConfig(FalconTask.b1), batch_size=batch_size)
        if batch_size != 1:
            raise ValueError("B1 ARTP decoder batch size must be 1")
        config.validate()
        self.payloads = payloads
        self.config = config
        self.active_date: Optional[str] = None
        self.query_label_access_count = 0
        self.model_updates = 0
        self.last_evidence: Optional[dict] = None

    def reset(self, dataset_tags: List = [""]):
        resolved = [resolve_dataset_tag(tag) for tag in dataset_tags]
        if len(set(resolved)) != 1:
            raise ValueError(f"conflicting dataset tags: {dataset_tags}")
        date = resolved[0]
        if date not in self.payloads:
            raise ValueError(f"no ARTP payload for {date}")
        self.active_date = date
        self.last_evidence = None
        return date

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        if self.active_date is None:
            raise RuntimeError("decoder.reset must select a date before predict")
        prediction, evidence = predict_from_payload(
            neural_observations,
            self.payloads[self.active_date],
            config=self.config,
            use_query_neural=True,
        )
        self.last_evidence = evidence
        return np.asarray(assert_predict_shape(prediction), dtype=np.float64)

    def on_done(self, dones: np.ndarray):
        return None
