# Full-block m30 activity-30 static M2 cached-identity submission image.
#
# Derives from the exact original-SPINT base image that already passed EvalAI
# M2 (same base as official submissions 578221/581359/581361/581362), so the
# conda env, falcon_challenge 1.0.2 pin, and NumPy 1.x runtime are the
# validated ones.  Only the payload and decoder module are added.
#
# Build (from THIS directory as the docker context):
#   docker build \
#     --build-arg PAYLOAD_SHA256=<payload sha256 from artifacts/*.receipt.json> \
#     -t spint-t4-m2:m30-act30-s42-<payload sha256 first 8> \
#     -f Dockerfile .
ARG BASE_IMAGE=spint-m2:e8-epoch027-76f0fb2
FROM ${BASE_IMAGE}

ARG PAYLOAD_SHA256=unknown
ARG CHECKPOINT_SHA256=25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e
ARG BASE_IMAGE_ID=sha256:b179efcba8e36202aec688b3104397919576e30d4344c309febcf692d4d7aaf8
ARG BASE_MODEL_SHA256=76f0fb2092c81b5fed94cf947f859f685b03be095ea48d2e85a64cde421be3b0
LABEL ai.eval.method="Full-block m30 activity-30 static M2 cached identity (chronological first-30 ridge T4 fit; B3S activity pool = first 30 label-free calibration trials)"
LABEL ai.eval.model.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.checkpoint.sha256="${CHECKPOINT_SHA256}"
LABEL ai.eval.base.image.id="${BASE_IMAGE_ID}"
LABEL ai.eval.base.model.sha256="${BASE_MODEL_SHA256}"
LABEL ai.eval.label_budget="30"
LABEL ai.eval.activity_budget="30"

COPY artifacts/t4_m2_seed42_m30_act30_identity.pkl /data/decoder.pkl
COPY act30_full_decoder.py /act30_full_decoder.py
COPY decode.py /decode.py

ENV EVALUATION_LOC=remote
ENV TASK=m2
ENV PHASE=test
ENV BATCH_SIZE=7

CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
