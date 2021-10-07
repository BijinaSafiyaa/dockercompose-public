# ==============================================================================
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
FROM openvino/ubuntu20_data_runtime
USER root
RUN apt-get update && apt-get -y upgrade && apt-get autoremove
COPY . /application
WORKDIR /application/example_client/
ENV http_proxy="http://proxy-chain.intel.com:911"
ENV https_proxy="http://proxy-chain.intel.com:912"
RUN pip3 install -r client_requirements.txt
RUN mkdir /application/example_client/results
ENV http_proxy=""
ENV https_proxy=""
ENTRYPOINT ["bash", "./run.sh"]
