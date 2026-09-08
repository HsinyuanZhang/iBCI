ARG BASE_IMAGE=spint-original-m1:e9-epoch019-052e9ea
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256
LABEL ai.eval.method="M1 RIFT joint D R100 concat e3 EMA cached CPU; architecture=RIFT backend=cached identity=live_b3s_concat_sealed; not BT-EORT; not ORT; not SPINT"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.task="m1"
LABEL ai.eval.old_spint_decoder="false"
LABEL ai.eval.exact_e="false"
LABEL ai.eval.architecture="RIFT"
LABEL ai.eval.backend="cached"
LABEL ai.eval.identity="live_b3s_concat_sealed"
LABEL ai.eval.dataloader_workers="0"
LABEL ai.eval.candidate="m1_rift_joint_d_e3_cached"
COPY artifacts/pkg /pkg
COPY artifacts/m1_rift_joint_d_e3.pkl /data/decoder.pkl
COPY m1_rift_falcon_decoder.py /m1_rift_falcon_decoder.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=remote TASK=m1 PHASE=test BATCH_SIZE=4 PYTHONUNBUFFERED=1 PYTHONPATH=/pkg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RIFT_PKG=/pkg
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
