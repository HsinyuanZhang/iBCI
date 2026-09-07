ARG BASE_IMAGE=spint-original-m1:e9-epoch019-052e9ea
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256=3e260ebdda7d0482094366a53daf6068ca48be9dd6660bd8b1e8f8738251dbfa
LABEL ai.eval.method="M1 proj_add P16 CausalPE4 (seed42 EMA e24) exact-E w0; endpoint24 EMA; official uint8/float coerce"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.task="m1"
LABEL ai.eval.label_budget="10"
LABEL ai.eval.exact_e="true"
LABEL ai.eval.proj_dim="16"
LABEL ai.eval.dataloader_workers="0"
LABEL ai.eval.torch_threads="2"
COPY artifacts/m1_projadd_p16_s42_ema_e24.pkl /data/decoder.pkl
COPY trf_falcon_decoder.py /trf_falcon_decoder.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=remote TASK=m1 PHASE=test BATCH_SIZE=4 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
