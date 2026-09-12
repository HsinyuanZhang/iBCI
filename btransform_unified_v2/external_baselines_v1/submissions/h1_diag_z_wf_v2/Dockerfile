FROM h1-epfilm-c1:no-readout-v1-523d3d2e
WORKDIR /workspace
COPY falcon_decoder.py decode.py /workspace/
COPY payload.npz manifest.json /data/
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
CMD ["/bin/bash", "-c", "python /workspace/decode.py --evaluation ${EVALUATION_LOC:-remote} --split ${TASK} --phase ${PHASE:-test} --batch-size ${BATCH_SIZE}"]

LABEL ai.eval.task="h1" ai.eval.method="fair-v2-diag_z_wf" ai.eval.payload.sha256="d64fcebf7be38dd652fff74a59b6781cda9d3156b9306be566db0ffd1cc5f544" ai.eval.decoder.sha256="2829d4287d99bf054c845051e10a3511cb343f313b94a268ecb01c1a13cfe43c" ai.eval.model.sha256="d64fcebf7be38dd652fff74a59b6781cda9d3156b9306be566db0ffd1cc5f544" ai.eval.checkpoint.sha256="" ai.eval.cpu_only="true" org.opencontainers.image.title="fair-v2-h1-diag_z_wf:cpu" org.opencontainers.image.description="Fair v2 CPU NumPy diag_z_wf Wiener-filter linear decoder" org.opencontainers.image.revision="fair-v2-linear"
ENV TASK=h1 BATCH_SIZE=8 PHASE=test EVALUATION_LOC=remote
LABEL ibci.h1.calibration.trials="" ibci.h1.checkpoint.sha256="" ibci.h1.epoch.zero_based="" ibci.h1.film_state.sha256="" ibci.h1.model_state.sha256="" ibci.h1.package.sha256="" ibci.h1.readout="" ibci.h1.selection.surface=""
