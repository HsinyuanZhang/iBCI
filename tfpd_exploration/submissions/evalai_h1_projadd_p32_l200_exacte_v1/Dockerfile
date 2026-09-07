ARG BASE_IMAGE=h1-epfilm-c1:no-readout-v1-523d3d2e
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256=ec806ff4d8a0f35d2569c7c4428cd6de2606dd4949ca2643b4ecc83757a70418
LABEL ai.eval.method="H1 proj_add P32 CausalPE4 stage-2 full-13 retrain (seed42 EMA e24 L=200) exact-E w0; not SPINT; not a C2 identity swap; not the P16 e24 image"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.task="h1"
LABEL ai.eval.old_spint_decoder="false"
LABEL ai.eval.exact_e="true"
LABEL ai.eval.dataloader_workers="0"
LABEL ai.eval.proj_dim="32"
LABEL ai.eval.token_in="36"
LABEL ai.eval.candidate="h1_projadd_p32_stage2_s42_ema_e24_L200_w0"
COPY artifacts/h1_projadd_p32_stage2_s42_ema_e24_L200.pkl /data/decoder.pkl
COPY h1_trf_falcon_decoder.py /h1_trf_falcon_decoder.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=remote TASK=h1 PHASE=test BATCH_SIZE=8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
