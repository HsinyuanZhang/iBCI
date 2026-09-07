FROM spint-m2:e8-epoch027-76f0fb2
WORKDIR /workspace
COPY evalai_m2_aof_scalar_static_v1/aofs_static_decoder.py evalai_m2_aof_scalar_static_v1/laws.py evalai_m2_aof_scalar_static_v1/decode.py /workspace/
COPY evalai_m2_aof_scalar_static_v2/artifacts/local_build_v2/t4_m2_seed42_dopt4_act30_aofs_identity.pkl /data/decoder.pkl
ENTRYPOINT ["python", "/workspace/decode.py"]
