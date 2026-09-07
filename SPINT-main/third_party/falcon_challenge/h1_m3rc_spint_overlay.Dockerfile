FROM 905418259932.dkr.ecr.us-east-1.amazonaws.com/few-shot-algorithms-for-consistent-neural-decoding-falcon-2319-participant-team-41975:13bdf438-9342-4f9d-94a0-5e74b1b771a7

# The base is the immutable local image used for the successful H1 all-source
# C1 submission family.  Reusing it preserves the already validated runtime
# and lets EvalAI reuse its existing registry layers.  The scientific payload
# and all route-owned code are replaced below.
ARG PACKAGE_SHA256
ARG CHECKPOINT_SHA256
ARG BATCH_SIZE=8

ADD ./local_data/h1_m3rc_evalai_v1/decoder.pt /data/decoder.pt
ADD ./third_party/ /third_party/
ADD ./src/ /src/
ADD ./third_party/falcon_challenge/h1_m3rc_spint_sample.py /decode.py

LABEL org.opencontainers.image.title="H1 M3RC SPINT" \
      org.opencontainers.image.description="Frozen C1/M3 carrier-aware H1 decoder with exactly-M3 MAT7 readout calibration" \
      ibci.h1.base.image.id="sha256:93ddcdb0213c43518ef67a6cc4ec32e0ee6ef416750f738ff4e5a845bc326f4b" \
      ibci.h1.package.sha256="${PACKAGE_SHA256}" \
      ibci.h1.checkpoint.sha256="${CHECKPOINT_SHA256}" \
      ibci.h1.calibration.trials="3"

ENV EVALUATION_LOC=remote \
    TASK=h1 \
    PHASE=test \
    BATCH_SIZE=${BATCH_SIZE} \
    EVAL_DATA_PATH=/dataset/evaluation_data \
    PREDICTION_PATH=/submission/submission.csv \
    PREDICTION_PATH_LOCAL=/tmp/submission.pkl \
    GT_PATH=/tmp/ground_truth.pkl

CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pt --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
