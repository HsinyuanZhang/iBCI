FROM spint-m2:e8-epoch027-76f0fb2
WORKDIR /workspace
COPY aofs_static_decoder.py laws.py /workspace/tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/
COPY decode.py /workspace/
COPY third_party/__init__.py /workspace/third_party/__init__.py
COPY third_party/falcon_challenge/__init__.py /workspace/third_party/falcon_challenge/__init__.py
COPY third_party/falcon_challenge/filtering.py /workspace/third_party/falcon_challenge/filtering.py
ENV PYTHONPATH=/workspace/tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1:/workspace
COPY decoder.pkl /data/decoder.pkl
ENTRYPOINT ["python", "/workspace/decode.py"]
