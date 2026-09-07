#!/usr/bin/env bash
# Environment for the cebra_exploration project.
#
#   source cebra_exploration/scripts/cebra_env.sh
#   "$CEBRA_PY" my_script.py
#
# PYTHONNOUSERSITE is mandatory: a user-site torch 2.12.0+cu130 shadows the env
# and reports cuda False because it is too new for driver 535.309.01. With user
# site disabled the env's torch 2.5.1.post303 sees both RTX 3090s.

CEBRA_REPO_ROOT="/home/xinyuan/Work_host/SPINT"
CEBRA_VENDOR="${CEBRA_REPO_ROOT}/cebra_exploration/third_party/cebra"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${CEBRA_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"
export CEBRA_PY="/home/xinyuan/miniconda3/envs/spint/bin/python"

# Installing packages: pypi.org through the clash proxy on 127.0.0.1:7890 gives
# intermittent SSL EOF. Strip the proxy and use the Tsinghua mirror instead, and
# always pass --no-deps unless the transitive set has been checked. Breaking the
# spint env breaks the paper's GPU pipeline.
cebra_pip_install() {
  env -u http_proxy -u https_proxy -u all_proxy \
      -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      PYTHONNOUSERSITE=1 "${CEBRA_PY}" -m pip install \
      --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple "$@"
}
