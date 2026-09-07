FROM spint-m2:e8-epoch027-76f0fb2
WORKDIR /workspace
COPY aofs_static_decoder.py laws.py decode.py /workspace/
COPY artifacts/local_build_v1/t4_m2_seed42_dopt4_act30_aofs_identity.pkl /data/decoder.pkl
ENTRYPOINT ["python", "/workspace/decode.py"]
