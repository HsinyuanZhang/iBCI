# Build from /home/xinyuan/Work_host/SPINT/sua_exploration so the validated
# runtime files can be reused without copying their implementation.
ARG BASE_IMAGE=spint-m2:e8-epoch027-76f0fb2
FROM ${BASE_IMAGE}

ARG PAYLOAD_FILE=evalai_t4_m2_activity_budget/artifacts/UNSET.pkl
ARG PAYLOAD_SHA256=unknown
ARG BUDGET=unknown
ARG CHECKPOINT_SHA256=25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e
LABEL ai.eval.method="ridge T4 M${BUDGET} activity30 cached identity"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.checkpoint.sha256="${CHECKPOINT_SHA256}"
LABEL ai.eval.label_budget="${BUDGET}"
LABEL ai.eval.activity_budget="30"

COPY ${PAYLOAD_FILE} /data/decoder.pkl
COPY evalai_t4_m2/t4_spint_decoder.py /t4_spint_decoder.py
COPY evalai_t4_m2/decode.py /decode.py

ENV EVALUATION_LOC=remote
ENV TASK=m2
ENV PHASE=test
ENV BATCH_SIZE=7

CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
