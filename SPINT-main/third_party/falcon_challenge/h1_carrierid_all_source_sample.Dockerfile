# Dedicated H1 all-source CarrierID recovery-candidate image.
#
# This file intentionally does not modify ``spint_sample.Dockerfile``.  The
# all-source runtime has a different entry point and payload contract: held-in
# records use four calibration trials and held-out records use three, while the neural checkpoint stays
# frozen at source-training epoch 49.  Build context is SPINT-main, for example:
#
#   docker build -f third_party/falcon_challenge/h1_carrierid_all_source_sample.Dockerfile \
#       --build-arg MODEL_FILE=h1_carrierid_all_source_payload.pkl \
#       --build-arg RECOVERY_AUDIT_SHA256=<sha256> \
#       --build-arg RECOVERY_PAYLOAD_SHA256=<sha256> \
#       --build-arg RECOVERY_STABILITY_SHA256=<canonical-v5-r4-sha256> \
#       -t spint_h1_all_source:epoch49 .
#
# Building/pushing this image is deliberately outside the source-only and
# recovery-audit gates.  The immutable recovery-package audit and payload SHA
# are supplied as build arguments so a future submission receipt can bind the
# image to those exact bytes; no private labels are copied into the image.

FROM condaforge/miniforge3

COPY environment.yaml /tmp/environment.yaml
RUN CONDA_OVERRIDE_CUDA="11.6" mamba env create -f /tmp/environment.yaml
RUN echo "mamba activate spint" > ~/.bashrc
ENV PATH=/opt/conda/envs/spint/bin:$PATH

# Match the audited FALCON evaluator and NumPy ABI used by the baseline image.
RUN /bin/bash -c "python3 -m pip install falcon_challenge==1.0.2"
RUN pip install "numpy<2"

ENV PREDICTION_PATH=/submission/submission.csv
ENV PREDICTION_PATH_LOCAL=/tmp/submission.pkl
ENV GT_PATH=/tmp/ground_truth.pkl
ENV EVALUATION_LOC=remote

# All-source runtime defaults.  The decoder is always embedded at the fixed
# path /data/decoder.pkl; the payload SHA is checked by the offline receipt and
# again by the future Docker/EvalAI preflight helper.
ARG TASK=h1
ARG BATCH_SIZE=8
ARG PHASE=test
ARG MODEL_FILE=h1_carrierid_all_source_payload.pkl
ARG RECOVERY_AUDIT_SHA256
ARG RECOVERY_PAYLOAD_SHA256
ARG RECOVERY_STABILITY_SHA256
ARG DECODER_SHA256

ADD ./local_data/${MODEL_FILE} /data/decoder.pkl
ADD ./third_party/ /third_party/
ADD ./src/ /src/
ADD ./third_party/falcon_challenge/h1_carrierid_all_source_sample.py /decode.py

# A tag or an omitted build argument is never submission-ready.  Fail the
# image build itself when the bytes copied into /data/decoder.pkl do not bind
# to both the recovery payload SHA and the explicitly supplied decoder SHA.
RUN test -n "$RECOVERY_AUDIT_SHA256" \
    && test -n "$RECOVERY_PAYLOAD_SHA256" \
    && test -n "$RECOVERY_STABILITY_SHA256" \
    && test -n "$DECODER_SHA256" \
    && test "$RECOVERY_AUDIT_SHA256" != "UNBOUND" \
    && test "$RECOVERY_PAYLOAD_SHA256" != "UNBOUND" \
    && test "$RECOVERY_STABILITY_SHA256" != "UNBOUND" \
    && test "$RECOVERY_STABILITY_SHA256" = "4b6adb58460a48cb791a56ed92efd1eb1f2ab961c5f43dd366777267da3a7277" \
    && test "$DECODER_SHA256" != "UNBOUND" \
    && printf '%s\n' "$RECOVERY_AUDIT_SHA256" | grep -Eq '^[0-9a-fA-F]{64}$' \
    && printf '%s\n' "$RECOVERY_PAYLOAD_SHA256" | grep -Eq '^[0-9a-fA-F]{64}$' \
    && printf '%s\n' "$RECOVERY_STABILITY_SHA256" | grep -Eq '^[0-9a-fA-F]{64}$' \
    && printf '%s\n' "$DECODER_SHA256" | grep -Eq '^[0-9a-fA-F]{64}$' \
    && observed=$(sha256sum /data/decoder.pkl | awk '{print $1}') \
    && test "$observed" = "$RECOVERY_PAYLOAD_SHA256" \
    && test "$observed" = "$DECODER_SHA256"

ENV TASK=${TASK}
ENV BATCH_SIZE=${BATCH_SIZE}
ENV PHASE=${PHASE}
ENV MODEL_FILE=${MODEL_FILE}
ENV RECOVERY_AUDIT_SHA256=${RECOVERY_AUDIT_SHA256}
ENV RECOVERY_PAYLOAD_SHA256=${RECOVERY_PAYLOAD_SHA256}
ENV RECOVERY_STABILITY_SHA256=${RECOVERY_STABILITY_SHA256}
ENV DECODER_SHA256=${DECODER_SHA256}
ENV EVAL_DATA_PATH=/dataset/evaluation_data

# Keep the command compatible with the FALCON evaluator's remote/local modes.
CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --phase $PHASE --batch-size $BATCH_SIZE"]
