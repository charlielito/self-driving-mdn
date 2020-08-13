import argparse
import base64
import io
import os
import shutil
import time
from datetime import datetime
from io import BytesIO

import cv2
import dicto
import eventlet
import eventlet.wsgi

import numpy as np
import socketio
import tensorflow as tf
import typer
from flask import Flask
from PIL import Image
from tensorflow_probability import distributions as tfd

from training.experiment import slice_parameter_vectors

# os.environ["CUDA_VISIBLE_DEVICES"] = ""


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.0
        self.error = 0.0
        self.integral = 0.0

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error only if has not exploded
        if abs(self.Ki * self.integral) < 100:
            self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


sio = socketio.Server()
app = Flask(__name__)
model = None

controller = SimplePIController(0.1, 0.002)

params = None
components = None


@sio.on("telemetry")
def telemetry(sid, data):
    global logs
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)

        image_array = image_array[params.crop_up : -params.crop_down, :, :]
        image_array = cv2.resize(image_array, tuple(params.image_size[::-1]))
        # image_array = (
        #     cv2.cvtColor(image_array, cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0
        # )
        image_array = image_array.astype(np.float32) / 255.0

        show_image = (255 * image_array[..., ::-1]).astype(np.uint8)

        preds = model(image=tf.constant(image_array[None, :, :, :]))

        if components is not None:
            steering_angle = get_steering(preds["pvec"])
        else:
            steering_angle = float(preds["steering"].numpy()[0])
        cv2.imshow("Visualizer", show_image)
        cv2.waitKey(1)

        throttle = controller.update(float(speed))

        # print(steering_angle, throttle)
        send_control(steering_angle, throttle)

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit("manual", data={}, skip_sid=True)


@sio.on("connect")
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            "steering_angle": steering_angle.__str__(),
            "throttle": throttle.__str__(),
        },
        skip_sid=True,
    )


def get_steering(preds):

    alpha, mu, sigma = slice_parameter_vectors(preds.numpy(), components)
    # print(alpha)
    max_prob = np.max(alpha, axis=-1)
    if max_prob > 0.9995:
        index = np.argmax(alpha, axis=-1)
        angle = mu[:, index[0]]
    else:
        angle = np.multiply(alpha, mu).sum(axis=-1)
    # print(angle)

    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.Normal(loc=mu, scale=sigma),
    )
    x = np.linspace(-1, 1, int(1e3))
    pyx = gm.prob(x)

    plot = cv2.plot.Plot2d_create(
        np.array(x).astype(np.float64), np.array(pyx).astype(np.float64)
    )
    plot.setPlotBackgroundColor((255, 255, 255))
    plot.setInvertOrientation(True)
    plot.setPlotLineColor(0)
    plot = plot.render()

    cv2.imshow("Distribution", plot)

    return angle[0]


def main(model_path: str, speed: float = 22):
    global app
    global model
    global params
    global components

    params = dicto.load(os.path.join(model_path, "params.yml"))
    components = params["components"]

    model_obj = tf.saved_model.load(model_path)
    model = model_obj.signatures["serving_default"]

    controller.set_desired(speed)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # depcloy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)


if __name__ == "__main__":
    typer.run(main)
