FROM spint-t4-m2:movement-t4-empty-s42-4e4dae8f

ARG PAYLOAD_SHA256
COPY artifacts/t4_m2_seed44_epoch08_movement_t4_empty_identity.pkl /artifacts/t4_m2_seed42_identity.pkl
LABEL org.opencontainers.image.title="M2 MOVE-T4 profile-free adapter epoch pick" \
      org.opencontainers.image.description="Movement-window T4 with profile-free adapter selected on visible held-out M2 data" \
      ai.eval.payload.sha256="${PAYLOAD_SHA256}" \
      ai.eval.candidate="movement_t4_m33_empty_adapter_seed44_epoch08_epochpick" \
      ai.eval.method="M2 MOVE-T4 plus profile-free adapter, visible-held-out epoch pick" \
      ibci.m2.payload.sha256="${PAYLOAD_SHA256}" \
      ibci.m2.adapter.head.sha256="13551d3fc33d1cc296670c577c11519c04abdb42fa081f7d061191bb5610e6c5" \
      ibci.m2.adapter.seed="44" \
      ibci.m2.adapter.epoch.one_based="8" \
      ibci.m2.selection.surface="visible-held-out-external-6"
