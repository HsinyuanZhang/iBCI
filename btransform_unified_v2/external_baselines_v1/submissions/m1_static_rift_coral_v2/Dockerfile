FROM spint-original-m1:e9-epoch019-052e9ea
ARG PAYLOAD_SHA256
WORKDIR /workspace
COPY artifacts/pkg /pkg
COPY artifacts/decoder.pkl /data/decoder.pkl
COPY static_falcon_decoder.py /workspace/static_falcon_decoder.py
COPY static_decode.py /workspace/decode.py
ENV EVALUATION_LOC=remote PHASE=test PYTHONUNBUFFERED=1 PYTHONPATH=/pkg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RIFT_PKG=/pkg CUDA_VISIBLE_DEVICES=""
CMD ["/bin/bash", "-c", "python /workspace/decode.py --evaluation ${EVALUATION_LOC:-remote} --model-path /data/decoder.pkl --split ${TASK} --phase ${PHASE:-test} --batch-size ${BATCH_SIZE}"]

LABEL ai.eval.task="m1" ai.eval.method="fair-v2-static_rift_coral" ai.eval.payload.sha256="99d5033ad7f61722445e638b8147f85d861ac0a0d107d8ec9f2131138c812923" ai.eval.cpu_only="true" ai.eval.architecture="RIFT" ai.eval.backend="cached" ai.eval.identity="static" org.opencontainers.image.title="fair-v2-m1-static_rift_coral:cpu"
ENV TASK=m1 BATCH_SIZE=4 PHASE=test EVALUATION_LOC=remote
