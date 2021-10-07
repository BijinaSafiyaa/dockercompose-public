from flask import Flask, render_template, Response, jsonify
import cv2
import time
# Initialize the Flask app
app = Flask(__name__)

outputFrame = None
params = None

def update_frame(in_frame, in_params):
    global outputFrame, params
   
    outputFrame = in_frame.copy()
    params = in_params


def gen_frames():
    global outputFrame
    while True:
        
        if outputFrame is None:
            continue
        #outputFrame = cv2.resize(outputFrame, (1280,720))
        ret, buffer = cv2.imencode('.jpg', outputFrame)
        if not ret:
            continue
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + bytearray(buffer) + b'\r\n')  # concat frame one by one and show result

@app.route('/update', methods = ['GET'])
def update():
    global params
    return jsonify(result=params)             

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def load_page():
    app.run(host='0.0.0.0', port=8000, threaded=True)

# if __name__ == "__main__":
#     app.run(host='0.0.0.0')