ARG BASE_IMAGE=spint-original-m1:e9-epoch019-052e9ea
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256
LABEL ai.eval.method="M1 RIFT NONE R100 concat selected EMA cached CPU; architecture=RIFT backend=cached identity=literal_zero_e0_t_none_concat"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.task="m1"
LABEL ai.eval.architecture="RIFT"
LABEL ai.eval.backend="cached"
LABEL ai.eval.identity="literal_zero_e0_t_none_concat"
LABEL ai.eval.arm="NONE"
LABEL ai.eval.dataloader_workers="0"
LABEL ai.eval.candidate="m1_rift_none_r100_selected_cached"
COPY artifacts/pkg /pkg
COPY artifacts/m1_rift_none_r100_selected.pkl /data/decoder.pkl
COPY m1_rift_none_falcon_decoder.py /m1_rift_none_falcon_decoder.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=remote TASK=m1 PHASE=test BATCH_SIZE=4 PYTHONUNBUFFERED=1 PYTHONPATH=/pkg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RIFT_PKG=/pkg
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
