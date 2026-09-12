ARG BASE_IMAGE=spint-original-m1:e9-epoch019-052e9ea
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256
ARG METHOD_LABEL
LABEL ai.eval.method="${METHOD_LABEL}"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.task="m1"
LABEL ai.eval.old_spint_decoder="false"
LABEL ai.eval.exact_e="false"
LABEL ai.eval.architecture="RIFT"
LABEL ai.eval.backend="cached"
LABEL ai.eval.identity="static"
LABEL ai.eval.recency="learned_slope_default"
LABEL ai.eval.dataloader_workers="0"
LABEL ai.eval.candidate="m1_rift_r100_static_e19_cached"
COPY artifacts/pkg /pkg
COPY artifacts/m1_rift_r100_static_e19.pkl /data/decoder.pkl
COPY m1_rift_falcon_decoder.py /m1_rift_falcon_decoder.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=remote TASK=m1 PHASE=test BATCH_SIZE=4 PYTHONUNBUFFERED=1 PYTHONPATH=/pkg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RIFT_PKG=/pkg CUDA_VISIBLE_DEVICES=""
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
