FROM spint-m2:e8-epoch027-76f0fb2
WORKDIR /workspace
COPY evalai_m2_aof_scalar_static_v1/aofs_static_decoder.py /workspace/aofs_static_decoder.py
COPY evalai_m2_aof_scalar_static_v2/artifacts/local_build_v2/t4_m2_seed42_dopt4_act30_aofs_identity.pkl /workspace/t4_m2_seed42_dopt4_act30_aofs_identity.pkl
ENTRYPOINT ["python", "/workspace/aofs_static_decoder.py"]
