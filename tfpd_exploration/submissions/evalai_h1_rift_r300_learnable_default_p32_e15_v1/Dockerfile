ARG BASE_IMAGE=h1-epfilm-c1:no-readout-v1-523d3d2e
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256
LABEL ai.eval.method="H1 RIFT R300 learned_slope default-ladder P32 e15 EMA cached CPU; architecture=RIFT backend=cached identity=signed_state14 recency=learned_slope_default; not BT-EORT; not ORT; not SPINT; not 582241; not 582196"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.task="h1"
LABEL ai.eval.old_spint_decoder="false"
LABEL ai.eval.exact_e="false"
LABEL ai.eval.architecture="RIFT"
LABEL ai.eval.backend="cached"
LABEL ai.eval.identity="signed_state14"
LABEL ai.eval.recency="learned_slope_default"
LABEL ai.eval.dataloader_workers="0"
LABEL ai.eval.candidate="h1_rift_r300_learned_slope_default_p32_e15_cached"
COPY artifacts/pkg /pkg
COPY artifacts/h1_rift_r300_learned_slope_default_p32_e15.pkl /data/decoder.pkl
COPY h1_rift_falcon_decoder.py /h1_rift_falcon_decoder.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=remote TASK=h1 PHASE=test BATCH_SIZE=8 PYTHONUNBUFFERED=1 PYTHONPATH=/pkg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RIFT_PKG=/pkg
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
