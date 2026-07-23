# SPINT decoder packaging - part of "SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding".
# Adapted from the FALCON challenge repo (https://github.com/snel-repo/falcon-challenge).
# Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Trung Le.

# Base image specifies basic dependencies; if you're using TF/Jax, you may want to use a different base image.
FROM condaforge/miniforge3

# Copy the environment.yaml file into the Docker image
COPY environment.yaml /tmp/environment.yaml
# Create the mamba environment from the environment.yaml file
RUN CONDA_OVERRIDE_CUDA="11.6" mamba env create -f /tmp/environment.yaml
# Activate the spint environment in all future bash shells
RUN echo "mamba activate spint" > ~/.bashrc
ENV PATH /opt/conda/envs/spint/bin:$PATH

RUN /bin/bash -c "python3 -m pip install falcon_challenge --upgrade"
RUN pip install "numpy<2"

ENV PREDICTION_PATH "/submission/submission.csv"
ENV PREDICTION_PATH_LOCAL "/tmp/submission.pkl"
ENV GT_PATH "/tmp/ground_truth.pkl"

# Users should install additional decoder-specific dependencies here.

ENV EVALUATION_LOC remote
# ENV EVALUATION_LOC local

# Add files from local context into Docker image
# Note local context reference is the working dir by default, see https://docs.docker.com/engine/reference/commandline/build/

# Build args — pass on the command line, e.g.
#   docker build --build-arg TASK=m1 --build-arg BATCH_SIZE=4 -t spint_m1:latest \
#                -f third_party/falcon_challenge/spint_sample.Dockerfile .
# Per-task BATCH_SIZE recommendations (passed via --build-arg BATCH_SIZE=...):
#   m1=4   m2=7   h1=8
# Defaults are M1's settings.
ARG TASK=m1
ARG BATCH_SIZE=4

# Add the packaged decoder for the chosen task.
# Note that Docker cannot easily import across symlinks; make sure data is not symlinked.
ADD ./local_data/spint_${TASK}.pkl data/decoder.pkl

# Add source code/configs
ADD ./third_party/ third_party/
ADD ./src/ src/

# Add runfile
ADD ./third_party/falcon_challenge/spint_sample.py decode.py

# Capture build-time ARGs into runtime ENVs so the CMD can reference them.
ENV TASK="${TASK}"
ENV BATCH_SIZE="${BATCH_SIZE}"
ENV PHASE="test"

# Make sure this matches the mounted data volume path. Generally leave as is.
ENV EVAL_DATA_PATH "/dataset/evaluation_data"

# CMD specifies a default command to run when the container is launched.
# It can be overridden with any cmd e.g. sudo docker run -it my_image /bin/bash


CMD ["/bin/bash", "-c", \
    "python decode.py --evaluation $EVALUATION_LOC --model-path data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]