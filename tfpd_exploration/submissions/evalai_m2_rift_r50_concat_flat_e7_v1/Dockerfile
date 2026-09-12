ARG BASE_IMAGE=spint-m2:e8-epoch027-76f0fb2
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256
ARG METHOD_LABEL
LABEL ai.eval.method="${METHOD_LABEL}"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.task="m2"
LABEL ai.eval.old_spint_decoder="false"
LABEL ai.eval.exact_e="false"
LABEL ai.eval.architecture="RIFT"
LABEL ai.eval.backend="cached"
LABEL ai.eval.identity="proj_add"
LABEL ai.eval.e0_source="concat_side_trunk"
LABEL ai.eval.recency="flat_zero_slope"
LABEL ai.eval.dataloader_workers="0"
LABEL ai.eval.candidate="m2_rift_r50_concat_flat_e7_cached"
COPY artifacts/pkg /pkg
COPY artifacts/m2_rift_r50_concat_flat_e7.pkl /data/decoder.pkl
COPY m2_rift_falcon_decoder.py /m2_rift_falcon_decoder.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=remote TASK=m2 PHASE=test BATCH_SIZE=7 PYTHONUNBUFFERED=1 PYTHONPATH=/pkg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RIFT_PKG=/pkg
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
