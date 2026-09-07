ARG BASE_IMAGE=spint-m2:e8-epoch027-76f0fb2
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256=4db109e75276d6e8f47df6540b0a4c8f6e955f81af1212127723976795f2eaa4
LABEL ai.eval.method="M2 SMALL concat CausalPE4 (S1-SMALL-COS seed42 EMA e8) ORT exact-E intra2; concat SMALL e8 ORT, not proj_add, not 581971"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.task="m2"
LABEL ai.eval.old_spint_decoder="false"
LABEL ai.eval.exact_e="true"
LABEL ai.eval.dataloader_workers="0"
LABEL ai.eval.ort="true"
LABEL ai.eval.identity="concat"
LABEL ai.eval.candidate="m2_small_concat_s42_ema_e8_ort"
LABEL ai.eval.label_budget="33"
LABEL ai.eval.activity_budget="33"
COPY artifacts/vendor/onnxruntime-1.19.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl /tmp/onnxruntime-1.19.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
RUN python -m pip install --no-index --no-deps /tmp/onnxruntime-1.19.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl && rm -f /tmp/onnxruntime-1.19.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
COPY artifacts/m2_small_trf_s42_ema_e08_ext6.pkl /data/decoder.pkl
COPY artifacts/ort_graphs /graphs
COPY trf_falcon_decoder.py /trf_falcon_decoder.py
COPY m2_concat_exacte_fast.py /m2_concat_exacte_fast.py
COPY m2_concat_exacte_ort.py /m2_concat_exacte_ort.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=remote TASK=m2 PHASE=test BATCH_SIZE=7 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 ORT_GRAPH_DIR=/graphs ORT_INTRA_OP=2 RT_PACKED_DECODER=/trf_falcon_decoder.py
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
