FROM h1-epfilm-c1:no-readout-v1-523d3d2e
ARG PAYLOAD_SHA256
WORKDIR /workspace
COPY artifacts/pkg /pkg
COPY artifacts/decoder.pkl /data/decoder.pkl
COPY static_falcon_decoder.py /workspace/static_falcon_decoder.py
COPY static_decode.py /workspace/decode.py
ENV EVALUATION_LOC=remote PHASE=test PYTHONUNBUFFERED=1 PYTHONPATH=/pkg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RIFT_PKG=/pkg CUDA_VISIBLE_DEVICES=""
CMD ["/bin/bash", "-c", "python /workspace/decode.py --evaluation ${EVALUATION_LOC:-remote} --model-path /data/decoder.pkl --split ${TASK} --phase ${PHASE:-test} --batch-size ${BATCH_SIZE}"]

LABEL ai.eval.task="h1" ai.eval.method="fair-v2-static_rift_coral" ai.eval.payload.sha256="1ac260fe423eb0cf00758fe846b2109885ccee19adab7a36676e14c404048b43" ai.eval.cpu_only="true" ai.eval.architecture="RIFT" ai.eval.backend="cached" ai.eval.identity="static" org.opencontainers.image.title="fair-v2-h1-static_rift_coral:cpu"
ENV TASK=h1 BATCH_SIZE=8 PHASE=test EVALUATION_LOC=remote
