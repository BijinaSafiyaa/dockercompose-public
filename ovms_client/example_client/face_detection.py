#
# Copyright (c) 2019-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import cv2
import time
import grpc
import numpy as np
import os
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from client_utils import print_statistics, prepare_certs
from common.FlaskPage import *
import threading


def preprocess(img, width, height):
    img = cv2.resize(img, (width, height))
    img = img.transpose(2,0,1).reshape(1,3,height,width)
    img = img.astype('float32')
    # change shape to NCHW
    return img

def main():
    parser = argparse.ArgumentParser(description='Demo for face detection requests via TFS gRPC API.'
                                                'analyses input images and saves with with detected faces.'
                                                'it relies on model face_detection...')

    parser.add_argument('--video_input', required=False, help='Directory with input images', default="videos/head-pose-face-detection-female-and-male.mp4")
    parser.add_argument('--output_dir', required=False, help='Directory for storing images with detection results', default="results")
    parser.add_argument('--batch_size', required=False, help='How many images should be grouped in one batch', default=1, type=int)
    parser.add_argument('--width', required=False, help='How the input image width should be resized in pixels', default=1200, type=int)
    parser.add_argument('--height', required=False, help='How the input image width should be resized in pixels', default=800, type=int)
    parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
    parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
    parser.add_argument('--model_name',required=False, default='face-detection', help='Specify the model name')
    parser.add_argument('--tls', default=False, action='store_true', help='use TLS communication with gRPC endpoint')
    parser.add_argument('--server_cert', required=False, help='Path to server certificate')
    parser.add_argument('--client_cert', required=False, help='Path to client certificate')
    parser.add_argument('--client_key', required=False, help='Path to client key')
    args = vars(parser.parse_args())

    address = "{}:{}".format(args['grpc_address'],args['grpc_port'])

    channel = None
    if args.get('tls'):
        server_ca_cert, client_key, client_cert = prepare_certs(server_cert=args['server_cert'],
                                                                client_key=args['client_key'],
                                                                client_ca=args['client_cert'])
        creds = grpc.ssl_channel_credentials(root_certificates=server_ca_cert,
                                            private_key=client_key, certificate_chain=client_cert)
        channel = grpc.secure_channel(address, creds)
    else:
        channel = grpc.insecure_channel(address)

    print("channel:", channel)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = args['model_name']
    cam = cv2.VideoCapture(args['video_input'])

    while True:
        
        start_time = time.time()
        ret, frame = cam.read()
        

        if ret == False:
            break

        height,width = frame.shape[:2]

        image = preprocess(frame, args["width"], args["height"])

        # send the input as protobuf
        request.inputs["data"].CopyFrom(make_tensor_proto(image, shape=None))
        
        result = stub.Predict(request, 10.0)
        output = make_ndarray(result.outputs["detection_out"])[0]
        #print(output.shape)
        # output = output[0].transpose(1,2,0)
        #print(output.shape)
        count = 0
        for i in range(0, 200-1):  # there is returned 200 detections for each image in the batch
            #print(output.shape)
            detection = output[:,i,:]
            #print(detection.shape)
            # each detection has shape 1,1,7 where last dimension represent:
            # image_id - ID of the image in the batch
            # label - predicted class ID
            # conf - confidence for the predicted class
            # (x_min, y_min) - coordinates of the top left bounding box corner
            #(x_max, y_max) - coordinates of the bottom right bounding box corner.
            if detection[0,2] > 0.5:  # ignore detections for image_id != y and confidence <0.5
                # print("detection", i , detection)
                x_min = int(detection[0,3] * width)
                y_min = int(detection[0,4] * height)
                x_max = int(detection[0,5] * width)
                y_max = int(detection[0,6] * height)
                # box coordinates are proportional to the image size
                #print("x_min", x_min)
                #print("y_min", y_min)
                #print("x_max", x_max)
                #print("y_max", y_max)
                count = count + 1

                img_out = cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,0,255),1)
                # draw each detected box on the input image
        end_time = time.time()
        stats = {}
        stats["no_of_face_detections"] = count
        stats["processing_time"] = end_time - start_time
        print("No of detections: ", count)
        print("Processing time: ", end_time - start_time)
        update_frame(img_out, stats)

if __name__ == '__main__':
    t = threading.Thread(target=load_page)
    t.daemon = True
    t.start()
    main()