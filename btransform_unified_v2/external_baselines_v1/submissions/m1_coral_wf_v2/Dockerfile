FROM spint-original-m1:e9-epoch019-052e9ea
WORKDIR /workspace
COPY falcon_decoder.py decode.py /workspace/
COPY payload.npz manifest.json /data/
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
CMD ["/bin/bash", "-c", "python /workspace/decode.py --evaluation ${EVALUATION_LOC:-remote} --split ${TASK} --phase ${PHASE:-test} --batch-size ${BATCH_SIZE}"]

LABEL ai.eval.task="m1" ai.eval.method="fair-v2-coral_wf" ai.eval.payload.sha256="a00529bd89ddd6604133718259213fb30d22d356b79ab3c1dff2438730915772" ai.eval.decoder.sha256="2829d4287d99bf054c845051e10a3511cb343f313b94a268ecb01c1a13cfe43c" ai.eval.model.sha256="a00529bd89ddd6604133718259213fb30d22d356b79ab3c1dff2438730915772" ai.eval.checkpoint.sha256="" ai.eval.cpu_only="true" org.opencontainers.image.title="fair-v2-m1-coral_wf:cpu" org.opencontainers.image.description="Fair v2 CPU NumPy coral_wf Wiener-filter linear decoder" org.opencontainers.image.revision="fair-v2-linear"
ENV TASK=m1 BATCH_SIZE=4 PHASE=test EVALUATION_LOC=remote
