FROM spint-original-m1:e9-epoch019-052e9ea
ARG PAYLOAD_SHA256
WORKDIR /workspace
COPY artifacts/pkg /pkg
COPY artifacts/decoder.pkl /data/decoder.pkl
COPY static_falcon_decoder.py /workspace/static_falcon_decoder.py
COPY static_decode.py /workspace/decode.py
ENV EVALUATION_LOC=remote PHASE=test PYTHONUNBUFFERED=1 PYTHONPATH=/pkg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RIFT_PKG=/pkg CUDA_VISIBLE_DEVICES=""
CMD ["/bin/bash", "-c", "python /workspace/decode.py --evaluation ${EVALUATION_LOC:-remote} --model-path /data/decoder.pkl --split ${TASK} --phase ${PHASE:-test} --batch-size ${BATCH_SIZE}"]

LABEL ai.eval.task="m1" ai.eval.method="fair-v2-static_rift_diag_z" ai.eval.payload.sha256="8d0f28ae139a5b22af59d13f7977088e7740b9f6946f739cd8dea9d64d53b2da" ai.eval.cpu_only="true" ai.eval.architecture="RIFT" ai.eval.backend="cached" ai.eval.identity="static" org.opencontainers.image.title="fair-v2-m1-static_rift_diag_z:cpu"
ENV TASK=m1 BATCH_SIZE=4 PHASE=test EVALUATION_LOC=remote
