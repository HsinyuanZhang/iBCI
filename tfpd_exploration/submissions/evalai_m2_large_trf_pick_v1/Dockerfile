ARG BASE_IMAGE=spint-m2:e8-epoch027-76f0fb2
FROM ${BASE_IMAGE}
ARG PAYLOAD_SHA256=4b46b4a27fb2df9bab1367ab46c0972558d7820988f1a03bef69001c85241411
ARG CHECKPOINT_SHA256=25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e
LABEL ai.eval.method="M2 large causal Transformer (B-TRANSFORMER seed42 RAW e20); architecture probe; visible-product pick on local ext-4; worst-session 0.2466 on 11-19; frozen M33 MOVE-T4/E0 banks; no TTA"
LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"
LABEL ai.eval.checkpoint.sha256="${CHECKPOINT_SHA256}"
LABEL ai.eval.label_budget="33"
LABEL ai.eval.activity_budget="33"
LABEL ai.eval.candidate="large_trf_s2_raw_e20"
LABEL ai.eval.old_spint_decoder="false"
COPY artifacts/m2_large_trf_s2_raw_e20.pkl /data/decoder.pkl
COPY trf_falcon_decoder.py /trf_falcon_decoder.py
COPY decode.py /decode.py
ENV EVALUATION_LOC=remote TASK=m2 PHASE=test BATCH_SIZE=7
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]
