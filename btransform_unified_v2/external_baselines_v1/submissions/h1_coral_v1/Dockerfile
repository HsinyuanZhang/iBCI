FROM h1-epfilm-c1:no-readout-v1-523d3d2e
WORKDIR /workspace
COPY falcon_decoder.py decode.py /workspace/
COPY payload.npz manifest.json /data/
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
CMD ["/bin/bash", "-c", "python /workspace/decode.py --evaluation ${EVALUATION_LOC:-remote} --split ${TASK} --phase ${PHASE:-test} --batch-size ${BATCH_SIZE}"]

LABEL ai.eval.task="h1" ai.eval.method="external-gf-coral" ai.eval.payload.sha256="72884cc50181990a98e1acb66245b8d24f4955dc32b15716ec15e05ad9e780c4" ai.eval.decoder.sha256="884350699865ab3ab1d58ff0565f7dff8fd67c96661b30f91a7f825813697b07" ai.eval.model.sha256="72884cc50181990a98e1acb66245b8d24f4955dc32b15716ec15e05ad9e780c4" ai.eval.checkpoint.sha256="" ai.eval.cpu_only="true" org.opencontainers.image.title="external-gf-h1-coral-v1" org.opencontainers.image.description="Frozen CPU NumPy coral plus Wiener FALCON decoder" org.opencontainers.image.revision="external-gf-runtime-v1"
ENV TASK=h1 BATCH_SIZE=8 PHASE=test EVALUATION_LOC=remote
LABEL ibci.h1.calibration.trials="" ibci.h1.checkpoint.sha256="" ibci.h1.epoch.zero_based="" ibci.h1.film_state.sha256="" ibci.h1.model_state.sha256="" ibci.h1.package.sha256="" ibci.h1.readout="" ibci.h1.selection.surface=""
