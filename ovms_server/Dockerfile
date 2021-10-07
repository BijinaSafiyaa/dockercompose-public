# ==============================================================================
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
FROM openvino/model_server
USER root
ENV http_proxy="http://proxy-chain.intel.com:911"
ENV https_proxy="http://proxy-chain.intel.com:912"
RUN curl --create-dirs https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/face-detection-retail-0004/FP32/face-detection-retail-0004.xml https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/face-detection-retail-0004/FP32/face-detection-retail-0004.bin -o /models/1/face-detection-retail-0004.xml -o /models/1/face-detection-retail-0004.bin
