FROM h1-epfilm-c1:no-readout-v1-523d3d2e

ARG PAYLOAD_SHA256
ARG MODEL_STATE_SHA256
ARG FILM_STATE_SHA256
COPY artifacts/decoder.pt /data/decoder.pt
LABEL org.opencontainers.image.title="H1 C2 held-out-selected epoch 15" \
      org.opencontainers.image.description="C2 M3-aware prefix-cycle H1 decoder selected on visible held-out development data; native identity, no fitted readout" \
      ibci.h1.package.sha256="${PAYLOAD_SHA256}" \
      ibci.h1.readout="identity-no-fit" \
      ibci.h1.checkpoint.sha256="ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215" \
      ibci.h1.model_state.sha256="${MODEL_STATE_SHA256}" \
      ibci.h1.film_state.sha256="${FILM_STATE_SHA256}" \
      ibci.h1.epoch.zero_based="15" \
      ibci.h1.selection.surface="visible-held-out-development" \
      ibci.h1.calibration.trials="3"
