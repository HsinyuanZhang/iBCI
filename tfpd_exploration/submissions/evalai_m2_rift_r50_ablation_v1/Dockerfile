ARG BASE_IMAGE=spint-m2:e8-epoch027-76f0fb2
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256
ARG ARM
LABEL ai.eval.task="m2" \
      ai.eval.architecture="RIFT-R50-concat" \
      ai.eval.backend="cached-pytorch-cpu" \
      ai.eval.identity="static-ablation" \
      ai.eval.arm="${ARM}" \
      ai.eval.payload.sha256="${PAYLOAD_SHA256}" \
      ai.eval.dataloader_workers="0" \
      ai.eval.method="M2 RIFT R50 concat selected-EMA static ACTIVITY_ONLY/NONE control"
COPY artifacts/pkg /pkg
COPY artifacts/m2_rift_r50_ablation.pkl /data/decoder.pkl
COPY m2_rift_falcon_decoder.py /m2_rift_falcon_decoder.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=remote TASK=m2 PHASE=test BATCH_SIZE=7 PYTHONUNBUFFERED=1 PYTHONPATH=/pkg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RIFT_PKG=/pkg
CMD ["/bin/bash","-c","python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
