FROM condaforge/miniforge3

COPY environment.yaml /tmp/environment.yaml
RUN CONDA_OVERRIDE_CUDA="11.8" mamba env create -f /tmp/environment.yaml
ENV PATH=/opt/conda/envs/spint/bin:$PATH
RUN python -m pip install --no-deps falcon-challenge==1.0.2 && pip install "numpy<2"

ARG PACKAGE_SHA256
ARG CHECKPOINT_SHA256
ARG BATCH_SIZE=8

ADD ./local_data/h1_m3rc_evalai_v1/decoder.pt /data/decoder.pt
ADD ./third_party/ /third_party/
ADD ./src/ /src/
ADD ./third_party/falcon_challenge/h1_m3rc_spint_sample.py /decode.py

LABEL org.opencontainers.image.title="H1 M3RC SPINT" \
      org.opencontainers.image.description="Frozen C1/M3 carrier-aware H1 decoder with exactly-M3 MAT7 readout calibration" \
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
