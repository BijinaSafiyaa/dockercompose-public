//*****************************************************************************
// Copyright 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include "dl_node.hpp"

#include <map>
#include <utility>

#include <inference_engine.hpp>

#include "dlnodesession.hpp"
#include "logging.hpp"
#include "modelmanager.hpp"
#include "ov_utils.hpp"
#include "ovinferrequestsqueue.hpp"
#include "prediction_service_utils.hpp"
#include "timer.hpp"

namespace ovms {

const uint WAIT_FOR_STREAM_ID_TIMEOUT_MICROSECONDS = 1;

Status DLNode::execute(session_key_t sessionKey, PipelineEventQueue& notifyEndQueue) {
    auto& nodeSession = getNodeSession(sessionKey);
    auto& dlNodeSession = static_cast<DLNodeSession&>(nodeSession);
    return dlNodeSession.execute(notifyEndQueue, WAIT_FOR_STREAM_ID_TIMEOUT_MICROSECONDS, *this);
}

Status DLNode::fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) {
    auto& dlNodeSession = static_cast<DLNodeSession&>(nodeSession);
    const auto& sessionMetadata = nodeSession.getNodeSessionMetadata();
    SessionResult sessionResults{sessionMetadata, {}};
    auto it = nodeSessionOutputs.emplace(sessionMetadata.getSessionKey(), std::move(sessionResults));
    if (!it.second) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Failed to put node: {} session: {} results in node session outputs",
            getName(), nodeSession.getSessionKey());
        return StatusCode::INTERNAL_ERROR;
    }
    auto& metadataBlobResultsPair = it.first->second;
    auto& blobResults = metadataBlobResultsPair.second;
    Status status;
    const uint waitTimeMicroseconds = 1;
    auto& inferRequest = dlNodeSession.getInferRequest(waitTimeMicroseconds);
    auto& model = dlNodeSession.getModelInstance();
    status = this->fetchResults(blobResults, inferRequest, model, nodeSession.getSessionKey());
    return status;
}

Status DLNode::fetchResults(BlobMap& outputs, InferenceEngine::InferRequest& inferRequest, ModelInstance& model, session_key_t sessionKey) {
    ReleaseSessionGuard releaseSessionGuard(this->getNodeSession(sessionKey));
    // Wait for blob results
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node: {} session: {} Waiting for infer request to finish", getName(), sessionKey);
    InferenceEngine::StatusCode ov_status;
    try {
        ov_status = inferRequest.Wait(InferenceEngine::IInferRequest::RESULT_READY);
    } catch (const InferenceEngine::Exception& e) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node: {} session: {} IE exception occured during infer request wait: {}", getName(), sessionKey, e.what());
        return StatusCode::INTERNAL_ERROR;
    } catch (std::exception& e) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node: {} session: {} exception occured during infer request wait: {}", getName(), sessionKey, e.what());
        return StatusCode::INTERNAL_ERROR;
    }
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node: {} session: {} infer request finished", getName(), sessionKey);
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Inference processing time for node {}; model name: {}; session: {} - {} ms",
        this->getName(),
        model.getName(),
        sessionKey,
        this->getNodeSession(sessionKey).getTimer().elapsed<std::chrono::microseconds>("inference") / 1000);

    static_cast<DLNodeSession&>(this->getNodeSession(sessionKey)).clearInputs();
    if (ov_status != InferenceEngine::StatusCode::OK) {
        Status status = StatusCode::OV_INTERNAL_INFERENCE_ERROR;
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node: {} session: {} Async infer failed: {}; OV StatusCode: {}", getName(), sessionKey, status.string(), ov_status);
        return status;
    }

    // Fill outputs map with result blobs. Fetch only those that are required in following nodes.
    for (const auto& node : this->next) {
        for (const auto& pair : node.get().getMappingByDependency(*this)) {
            const auto& output_name = pair.first;
            if (outputs.find(output_name) != outputs.end()) {
                continue;
            }

            try {
                std::string realModelOutputName;
                if (!getRealOutputName(model, output_name, &realModelOutputName).ok()) {
                    SPDLOG_LOGGER_WARN(dag_executor_logger, "Node: {} session: {} Cannot find real model output name for alias: {}", getName(), sessionKey, output_name);
                    return StatusCode::INTERNAL_ERROR;
                }
                SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node: {} session: {} Getting blob from model: {}, inferRequestStreamId: {}, blobName: {}",
                    getName(), sessionKey, modelName, sessionKey, realModelOutputName);
                const auto blob = inferRequest.GetBlob(realModelOutputName);
                SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node: {} session: {} Creating copy of blob from model: {}, blobName: {}",
                    getName(), sessionKey, modelName, realModelOutputName);
                InferenceEngine::Blob::Ptr copiedBlob;
                auto status = blobClone(copiedBlob, blob);
                if (!status.ok()) {
                    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Could not clone result blob; node: {}; session: {}; model name: {}; output: {}",
                        getName(),
                        this->modelName,
                        realModelOutputName);
                    return status;
                }
                outputs.emplace(std::make_pair(output_name, std::move(copiedBlob)));
            } catch (const InferenceEngine::Exception& e) {
                Status status = StatusCode::OV_INTERNAL_SERIALIZATION_ERROR;
                SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node: {} session:{} Error during getting blob {}; exception message: {}", getName(), sessionKey, status.string(), e.what());
                return status;
            }
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node: {} session: {} Blob with name {} has been prepared", getName(), sessionKey, output_name);
        }
    }
    return StatusCode::OK;
}

void DLNode::release(session_key_t sessionId) {
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Release node: {} sessionKey: {}", getName(), sessionId);
    getNodeSession(sessionId).release();
}
bool DLNode::tryDisarm(const session_key_t& sessionKey, const uint microseconds) {
    return getNodeSession(sessionKey).tryDisarm(microseconds);
}

std::unique_ptr<NodeSession> DLNode::createNodeSession(const NodeSessionMetadata& metadata, const CollapseDetails& collapsingDetails) {
    return std::make_unique<DLNodeSession>(metadata, getName(), previous.size(), collapsingDetails,
        this->modelManager, this->modelName, this->modelVersion.value_or(0));
}

}  // namespace ovms
