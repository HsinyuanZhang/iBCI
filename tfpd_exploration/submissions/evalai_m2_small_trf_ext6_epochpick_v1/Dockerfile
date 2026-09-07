ARG BASE_IMAGE=spint-m2:e8-epoch027-76f0fb2
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256=4db109e75276d6e8f47df6540b0a4c8f6e955f81af1212127723976795f2eaa4
ARG CHECKPOINT_SHA256=25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e
LABEL ai.eval.method="M2 small causal Transformer (S1-SMALL-COS seed42 EMA e8) exact-E w0; 6-session official held-out-calib epoch-pick; not 581971"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.checkpoint.sha256="${CHECKPOINT_SHA256}"
LABEL ai.eval.label_budget="33"
LABEL ai.eval.activity_budget="33"
LABEL ai.eval.candidate="small_trf_s42_ema_e8_ext6_w0"
LABEL ai.eval.old_spint_decoder="false"
LABEL ai.eval.exact_e="true"
LABEL ai.eval.dataloader_workers="0"
COPY artifacts/m2_small_trf_s42_ema_e08_ext6.pkl /data/decoder.pkl
COPY trf_falcon_decoder.py /trf_falcon_decoder.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=remote TASK=m2 PHASE=test BATCH_SIZE=7 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
