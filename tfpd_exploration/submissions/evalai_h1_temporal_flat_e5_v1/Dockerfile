ARG BASE_IMAGE=h1-epfilm-c1:no-readout-v1-523d3d2e
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256=91b9ace7640d8e0a75da1d79dcc566567d909f5bd2130626f475966c212d1cdb
ARG CALIBRATION_SHA256=ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215
LABEL ai.eval.method="H1 new temporal Transformer FLAT; snapshot epoch 5 of intended 12; EMA; known-source development; not C2 identity swap"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.calibration.sha256="${CALIBRATION_SHA256}"
LABEL ai.eval.old_c2_decoder="false"
LABEL ai.eval.c2_identity_swap="false"
LABEL ai.eval.epoch="5"
LABEL ai.eval.epochs_target="12"
LABEL ai.eval.view="EMA"
LABEL ai.eval.candidate="h1_temporal_flat_ema_e5"
LABEL ai.eval.decoder="h1-temporal-trf"
COPY artifacts/h1_temporal_flat_ema_e5.pkl /data/decoder.pkl
COPY h1_trf_falcon_decoder.py /h1_trf_falcon_decoder.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=remote TASK=h1 PHASE=test BATCH_SIZE=8
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
