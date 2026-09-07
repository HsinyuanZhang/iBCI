# Ce-NAT chunk100e continual TTA M2 submission image (exploratory, TTA track).
#
# Same validated base image as the scored act30 family. The decoder runs the
# energy-gated chunk memory online: seed = first-30 calibration block, commits
# gated by a label-free running-median rate rule, identity recomputed per
# commit. Declared IsTestTimeAdaptive=true.
#
# Build (from THIS directory as the docker context):
#   docker build \
#     --build-arg PAYLOAD_SHA256=<payload sha256 from receipt> \
#     -t spint-t4-m2:cenat-chunk100e-s42-<payload sha256 first 8> \
#     -f Dockerfile .
ARG BASE_IMAGE=spint-m2:e8-epoch027-76f0fb2
FROM ${BASE_IMAGE}

ARG PAYLOAD_SHA256=unknown
ARG CHECKPOINT_SHA256=c5672a2bfc94f55729d64e5eeb2f6917162c345713e7102dec4ad371cd147ea2
ARG BASE_IMAGE_ID=sha256:b179efcba8e36202aec688b3104397919576e30d4344c309febcf692d4d7aaf8
ARG BASE_MODEL_SHA256=76f0fb2092c81b5fed94cf947f859f685b03be095ea48d2e85a64cde421be3b0
LABEL ai.eval.method="Ce-NAT chunk100e continual TTA (matched joint-trained checkpoint; seed=first-30 calibration block; label-free energy-gated 100-bin chunk commits, running-median rule, capacity 30; identity recomputed per commit; decode-before-commit)"
LABEL ai.eval.model.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.checkpoint.sha256="${CHECKPOINT_SHA256}"
LABEL ai.eval.base.image.id="${BASE_IMAGE_ID}"
LABEL ai.eval.base.model.sha256="${BASE_MODEL_SHA256}"
LABEL ai.eval.label_budget="4"
LABEL ai.eval.activity_budget="30"
LABEL ai.eval.test_time_adaptive="true"

COPY artifacts/t4_m2_seed42_cenat_chunk100e_tta.pkl /data/decoder.pkl
COPY cenat_chunk_decoder.py /cenat_chunk_decoder.py
COPY decode.py /decode.py

ENV EVALUATION_LOC=remote
ENV TASK=m2
ENV PHASE=test
ENV BATCH_SIZE=7

CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
