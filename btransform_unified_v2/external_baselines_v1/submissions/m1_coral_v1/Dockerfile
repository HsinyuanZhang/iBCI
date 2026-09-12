FROM spint-original-m1:e9-epoch019-052e9ea
WORKDIR /workspace
COPY falcon_decoder.py decode.py /workspace/
COPY payload.npz manifest.json /data/
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
CMD ["/bin/bash", "-c", "python /workspace/decode.py --evaluation ${EVALUATION_LOC:-remote} --split ${TASK} --phase ${PHASE:-test} --batch-size ${BATCH_SIZE}"]

LABEL ai.eval.task="m1" ai.eval.method="external-gf-coral" ai.eval.payload.sha256="3ee7acbd09e8539cb2716907603cf837b4535ad056e2c9c32cfc9d77212da6a3" ai.eval.decoder.sha256="884350699865ab3ab1d58ff0565f7dff8fd67c96661b30f91a7f825813697b07" ai.eval.model.sha256="3ee7acbd09e8539cb2716907603cf837b4535ad056e2c9c32cfc9d77212da6a3" ai.eval.checkpoint.sha256="" ai.eval.cpu_only="true" org.opencontainers.image.title="external-gf-m1-coral-v1" org.opencontainers.image.description="Frozen CPU NumPy coral plus Wiener FALCON decoder" org.opencontainers.image.revision="external-gf-runtime-v1"
ENV TASK=m1 BATCH_SIZE=4 PHASE=test EVALUATION_LOC=remote
