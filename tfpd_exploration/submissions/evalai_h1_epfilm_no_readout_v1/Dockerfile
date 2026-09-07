FROM h1-epfilm-c1:evalai-v1-df71cb93

ARG PAYLOAD_SHA256
COPY artifacts/decoder.pt /data/decoder.pt
LABEL org.opencontainers.image.title="H1 EP-FILM without MAT7 readout" \
      org.opencontainers.image.description="Frozen all-source C1/M3 H1 EP-FILM decoder; exact identity output map" \
      ibci.h1.package.sha256="${PAYLOAD_SHA256}" \
      ibci.h1.readout="identity-no-fit" \
      ibci.h1.checkpoint.sha256="0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06" \
      ibci.h1.film_state.sha256="b602c2e090f455cb6259fc76fb9a225bfdb9aa6bf40adcd5ccd10b9a0da13b37" \
      ibci.h1.calibration.trials="3"
