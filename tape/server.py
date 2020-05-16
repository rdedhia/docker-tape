"""
Main entrypoint for the Flask webserver.
"""
from flask import (
    Flask,
    jsonify,
)
import torch


def health_check():
    """
    Shallow health check. Default status to 200 with response of 'Health Check OK'
    """
    app.logger.debug("Health check route")
    return "Health Check OK", 200


def get_torch_gpu_settings():
    """
    Return information about GPUs torch has access to

    Returns:
        response (flask.Response)
    """
    app.logger.debug("GPU settings route")
    response = {}
    if torch.cuda.is_available():
        response["cuda"] = True
        device_num = torch.cuda.current_device()
        response["device_count"] = torch.cuda.device_count()
        response["device_name"] = torch.cuda.get_device_name(device_num)
    else:
        response["cuda"] = False

    response = jsonify(response)
    response.status_code = 200

    return response


def create_app():
    """
    Main function to create the Flask app
    """
    flask_app = Flask(__name__)

    flask_app.add_url_rule(rule="/health/full", view_func=health_check, methods=["GET"])
    flask_app.add_url_rule(rule="/torch/gpu", view_func=get_torch_gpu_settings, methods=["GET"])

    return flask_app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8443)
