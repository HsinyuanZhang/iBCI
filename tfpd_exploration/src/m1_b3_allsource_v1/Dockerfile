# All-source B3 / B3S-rSyn3 cached-identity M1 image.
#
# Same pattern as the successful M2 cached-identity submission 581644:
# start from an already-accepted FALCON image, replace only payload + decoder.
ARG BASE_IMAGE=spint-original-m1:e9-epoch019-052e9ea
FROM ${BASE_IMAGE}

ARG PAYLOAD_SHA256=unknown
ARG CHECKPOINT_SHA256=unknown
ARG ARM=b3
ARG METHOD_LABEL=M1 all-source B3 cached identity chronological M10
ARG TEACHER_SHA256=c81a2bbd860452e6186a9ecf55c0b747da61baef4fae3212f61521be68cc5ac2
ARG BASE_IMAGE_ID=sha256:f5af9eb29b7f86616d898261070193b1b0777db75567848c62d7f888ce3d76cd
LABEL ai.eval.task="m1"
LABEL ai.eval.arm="${ARM}"
LABEL ai.eval.method="${METHOD_LABEL}"
LABEL ai.eval.model.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.checkpoint.sha256="${CHECKPOINT_SHA256}"
LABEL ai.eval.teacher.sha256="${TEACHER_SHA256}"
LABEL ai.eval.base.image.id="${BASE_IMAGE_ID}"

COPY artifacts/decoder.pkl /data/decoder.pkl
COPY runtime.py /runtime.py
COPY decode.py /decode.py

ENV EVALUATION_LOC=remote
ENV TASK=m1
ENV PHASE=test
ENV BATCH_SIZE=4
ENV ARM="${ARM}"

CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
