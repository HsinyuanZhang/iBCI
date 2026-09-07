"""Official-shape B1 decoder for frozen ARTP-P payloads."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.interface import BCIDecoder

from tfpd_exploration.src.b1_artp_v1.core import ARTPContractError
from tfpd_exploration.src.b1_sfcj_v1.decoder import assert_predict_shape, resolve_dataset_tag

from .core import ARTPProfileConfig, predict_from_profiled_payload


class B1ARTPProfileDecoder(BCIDecoder):
    def __init__(self, payloads: dict, *, config=ARTPProfileConfig(), task_config: Optional[FalconConfig] = None, batch_size: int = 1):
        super().__init__(task_config or FalconConfig(FalconTask.b1), batch_size=batch_size)
        if batch_size != 1:
            raise ARTPContractError("B1 ARTP-P requires batch_size=1")
        config.validate()
        self.payloads = payloads
        self.config = config
        self.active_date = None
        self.last_evidence = None
        self.query_label_access_count = 0
        self.model_updates = 0

    def reset(self, dataset_tags: List = [""]):
        dates = [resolve_dataset_tag(tag) for tag in dataset_tags]
        if len(set(dates)) != 1 or dates[0] not in self.payloads:
            raise ARTPContractError(f"invalid/missing payload tags: {dataset_tags}")
        self.active_date = dates[0]
        self.last_evidence = None
        return self.active_date

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        if self.active_date is None:
            raise ARTPContractError("reset must be called before predict")
        prediction, evidence = predict_from_profiled_payload(
            neural_observations, self.payloads[self.active_date], config=self.config
        )
        self.last_evidence = evidence
        return np.asarray(assert_predict_shape(prediction), dtype=np.float64)

    def on_done(self, dones: np.ndarray):
        return None
