ARG BASE_IMAGE=spint-original-m1:e9-epoch019-052e9ea
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256=5e44fc37965640138bf977da42a988e7636daa996d1eb9b50a75fa5628f9d7b3
LABEL ai.eval.method="M1 V3 isolated current-query diagnostic candidate; local only"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.local_only="true"
COPY artifacts/m1_optimized_v2_t_ema_e6.pkl /data/decoder.pkl
COPY m1_trf_falcon_decoder.py /m1_trf_falcon_decoder.py
COPY v3_runtime.py /v3_runtime.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=local TASK=m1 PHASE=test BATCH_SIZE=4 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
