ARG BASE_IMAGE=h1-epfilm-c1:no-readout-v1-523d3d2e
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256=91bb23f50b214b3841f19d6fcc8e7ac11c7b2b179f615b08be111f07dd4d1565
LABEL ai.eval.method="H1 C2-CAL-1 B2 proj_add CausalPE4 (seed42 EMA e18 L=200 P16) ORT exact-E intra2; not SPINT; not C2-identical; not 582025 e24"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.task="h1"
LABEL ai.eval.old_spint_decoder="false"
LABEL ai.eval.exact_e="true"
LABEL ai.eval.dataloader_workers="0"
LABEL ai.eval.ort="true"
LABEL ai.eval.candidate="h1_c2_cal1_b2_s42_ema_e18_L200_ort"
COPY artifacts/vendor/onnxruntime-1.19.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl /tmp/onnxruntime-1.19.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
RUN python -m pip install --no-index --no-deps /tmp/onnxruntime-1.19.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl && rm -f /tmp/onnxruntime-1.19.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
COPY artifacts/h1_c2_cal1_b2_s42_ema_e18_L200.pkl /data/decoder.pkl
COPY artifacts/ort_graphs /graphs
COPY h1_trf_falcon_decoder.py /h1_trf_falcon_decoder.py
COPY h1_exacte_fast.py /h1_exacte_fast.py
COPY h1_exacte_ort.py /h1_exacte_ort.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=remote TASK=h1 PHASE=test BATCH_SIZE=8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 ORT_GRAPH_DIR=/graphs ORT_INTRA_OP=2 RT_PACKED_DECODER=/h1_trf_falcon_decoder.py
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
