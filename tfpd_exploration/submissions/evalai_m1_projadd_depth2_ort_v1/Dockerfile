ARG BASE_IMAGE=spint-original-m1:e9-epoch019-052e9ea
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256=18eaa3c98da87760a835bb52d663e0e0048c3ebeec5396f960b12c18aeba4d6a
LABEL ai.eval.method="M1 proj_add P16 CausalPE depth-2 (seed42 EMA e21 L=100) ORT exact-E intra2; not SPINT; not 582019 depth-4"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.task="m1"
LABEL ai.eval.label_budget="10"
LABEL ai.eval.exact_e="true"
LABEL ai.eval.proj_dim="16"
LABEL ai.eval.dataloader_workers="0"
LABEL ai.eval.torch_threads="2"
LABEL ai.eval.ort="true"
LABEL ai.eval.temporal_layers="2"
COPY artifacts/vendor/onnxruntime-1.19.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl /tmp/onnxruntime-1.19.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
RUN python -m pip install --no-index --no-deps /tmp/onnxruntime-1.19.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl && rm -f /tmp/onnxruntime-1.19.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
COPY artifacts/m1_projadd_p16_d2_s42_ema_e21.pkl /data/decoder.pkl
COPY artifacts/ort_graphs /graphs
COPY trf_falcon_decoder.py /trf_falcon_decoder.py
COPY m1_exacte_fast.py /m1_exacte_fast.py
COPY m1_exacte_ort.py /m1_exacte_ort.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=remote TASK=m1 PHASE=test BATCH_SIZE=4 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 ORT_GRAPH_DIR=/graphs ORT_INTRA_OP=2 RT_SUBMITTED_DECODER=/trf_falcon_decoder.py
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
