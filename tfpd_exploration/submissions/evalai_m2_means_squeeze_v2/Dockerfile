# Native first-33 T4 + hold/reach mean-rate FiLM cached-identity M2 image.
#
# Same validated original-SPINT base as official 578221. Only the payload and
# decoder module are added. FiLM is offline; the runtime serves cached E[N,50].
ARG BASE_IMAGE=spint-m2:e8-epoch027-76f0fb2
FROM ${BASE_IMAGE}

ARG PAYLOAD_SHA256=0cc4ab769c8752191c87b6d8947b32768e15c51a49a097c3b6335b416fa81ca5
ARG CHECKPOINT_SHA256=25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e
ARG BASE_IMAGE_ID=sha256:b179efcba8e36202aec688b3104397919576e30d4344c309febcf692d4d7aaf8
ARG BASE_MODEL_SHA256=76f0fb2092c81b5fed94cf947f859f685b03be095ea48d2e85a64cde421be3b0
LABEL ai.eval.method="Native first-33 T4 plus hold-vs-reach mean-rate FiLM cached identity (activity pool = first 33; FiLM offline on held-in only; online cached E[N,50]; no TTA)"
LABEL ai.eval.model.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.checkpoint.sha256="${CHECKPOINT_SHA256}"
LABEL ai.eval.base.image.id="${BASE_IMAGE_ID}"
LABEL ai.eval.base.model.sha256="${BASE_MODEL_SHA256}"
LABEL ai.eval.label_budget="33"
LABEL ai.eval.activity_budget="33"
LABEL ai.eval.candidate="means_ep12_lr3e4_s42"

COPY artifacts/t4_m2_seed42_means_m33_ep12_lr3e4_identity.pkl /data/decoder.pkl
COPY t4_spint_decoder.py /t4_spint_decoder.py
COPY decode.py /decode.py

ENV EVALUATION_LOC=remote
ENV TASK=m2
ENV PHASE=test
ENV BATCH_SIZE=7

CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
