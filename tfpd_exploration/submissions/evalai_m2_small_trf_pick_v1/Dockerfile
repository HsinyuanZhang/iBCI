ARG BASE_IMAGE=spint-m2:e8-epoch027-76f0fb2
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256=37071f3e718913f994b5015e13aa203026fb630339e87921ddb1c35de9d2bf62
ARG CHECKPOINT_SHA256=25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e
LABEL ai.eval.method="M2 small causal Transformer (S1-SMALL-COS seed42 EMA e19); visible-product pick on local ext-4; frozen M33 MOVE-T4/E0 banks; no TTA"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.checkpoint.sha256="${CHECKPOINT_SHA256}"
LABEL ai.eval.label_budget="33"
LABEL ai.eval.activity_budget="33"
LABEL ai.eval.candidate="small_trf_s1_ema_e19"
LABEL ai.eval.old_spint_decoder="false"
COPY artifacts/m2_small_trf_s1_ema_e19.pkl /data/decoder.pkl
COPY trf_falcon_decoder.py /trf_falcon_decoder.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=remote TASK=m2 PHASE=test BATCH_SIZE=7
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
