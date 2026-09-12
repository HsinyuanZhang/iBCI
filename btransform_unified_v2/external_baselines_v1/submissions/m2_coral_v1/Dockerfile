FROM spint-m2:e8-epoch027-76f0fb2
WORKDIR /workspace
COPY falcon_decoder.py decode.py /workspace/
COPY payload.npz manifest.json /data/
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
CMD ["/bin/bash", "-c", "python /workspace/decode.py --evaluation ${EVALUATION_LOC:-remote} --split ${TASK} --phase ${PHASE:-test} --batch-size ${BATCH_SIZE}"]

LABEL ai.eval.task="m2" ai.eval.method="external-gf-coral" ai.eval.payload.sha256="647edb1844696e33e7ef917882150588ab2c09c15696a8c14e8780fac3435fae" ai.eval.decoder.sha256="884350699865ab3ab1d58ff0565f7dff8fd67c96661b30f91a7f825813697b07" ai.eval.model.sha256="647edb1844696e33e7ef917882150588ab2c09c15696a8c14e8780fac3435fae" ai.eval.checkpoint.sha256="" ai.eval.cpu_only="true" org.opencontainers.image.title="external-gf-m2-coral-v1" org.opencontainers.image.description="Frozen CPU NumPy coral plus Wiener FALCON decoder" org.opencontainers.image.revision="external-gf-runtime-v1"
ENV TASK=m2 BATCH_SIZE=7 PHASE=test EVALUATION_LOC=remote
