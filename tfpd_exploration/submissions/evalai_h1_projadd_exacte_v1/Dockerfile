ARG BASE_IMAGE=h1-epfilm-c1:no-readout-v1-523d3d2e
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256=d915af0a95dfe11021819333b13bd04f9ac8b91a6c176111986627acd74c97d7
LABEL ai.eval.method="H1 proj_add CausalPE4 stage-2 full-13 retrain (seed42 EMA e24 L=200) exact-E w0; not SPINT; not a C2 identity swap"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.task="h1"
LABEL ai.eval.old_spint_decoder="false"
LABEL ai.eval.exact_e="true"
LABEL ai.eval.dataloader_workers="0"
LABEL ai.eval.candidate="h1_projadd_stage2_s42_ema_e24_L200_w0"
COPY artifacts/h1_projadd_stage2_s42_ema_plus8_L200.pkl /data/decoder.pkl
COPY h1_trf_falcon_decoder.py /h1_trf_falcon_decoder.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=remote TASK=h1 PHASE=test BATCH_SIZE=8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
