ARG BASE_IMAGE=spint-m2:e8-epoch027-76f0fb2
FROM ${BASE_IMAGE}

ARG PAYLOAD_SHA256
ARG CHECKPOINT_SHA256=25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e
LABEL ai.eval.method="M2 movement-window T4 cached identity (M33, bins 5:30, no FiLM, no TTA)"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.checkpoint.sha256="${CHECKPOINT_SHA256}"
LABEL ai.eval.label_budget="33"
LABEL ai.eval.activity_budget="33"
LABEL ai.eval.candidate="movement_t4_bins5_30_m33"

COPY artifacts/t4_m2_seed42_movement_t4_identity.pkl /data/decoder.pkl
COPY t4_spint_decoder.py /t4_spint_decoder.py
COPY decode.py /decode.py

ENV EVALUATION_LOC=remote TASK=m2 PHASE=test BATCH_SIZE=7
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
