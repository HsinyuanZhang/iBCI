FROM spint-m2:e8-epoch027-76f0fb2
ARG PAYLOAD_SHA256
WORKDIR /workspace
COPY artifacts/pkg /pkg
COPY artifacts/decoder.pkl /data/decoder.pkl
COPY static_falcon_decoder.py /workspace/static_falcon_decoder.py
COPY static_decode.py /workspace/decode.py
ENV EVALUATION_LOC=remote PHASE=test PYTHONUNBUFFERED=1 PYTHONPATH=/pkg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RIFT_PKG=/pkg CUDA_VISIBLE_DEVICES=""
CMD ["/bin/bash", "-c", "python /workspace/decode.py --evaluation ${EVALUATION_LOC:-remote} --model-path /data/decoder.pkl --split ${TASK} --phase ${PHASE:-test} --batch-size ${BATCH_SIZE}"]

LABEL ai.eval.task="m2" ai.eval.method="fair-v2-static_rift_diag_z" ai.eval.payload.sha256="444bfd56add3dd52f5eb3cbd2a2eab5a1d7c187e01e894013220816033351dd5" ai.eval.cpu_only="true" ai.eval.architecture="RIFT" ai.eval.backend="cached" ai.eval.identity="static" org.opencontainers.image.title="fair-v2-m2-static_rift_diag_z:cpu"
ENV TASK=m2 BATCH_SIZE=7 PHASE=test EVALUATION_LOC=remote
