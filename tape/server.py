"""
Main entrypoint for the Flask webserver.
"""
from flask import (
    Flask,
    jsonify,
    request,
    send_file
)
import torch
import gzip
import os


def embed_data():
    """
    Return embedded data based on chosen model

    Returns:
        Embedded npz file from original gzipped fasta file
    
    Parameters:
        model:    transformer (default), unirep, trrosetta, onehot
        pretrained_model: bert-base (transformer, default), babbler-1900 (unirep), xaa, xab, xac, xad, xae (trRosetta)
        tokenizer: ipuac (default), unirep (unirep)
        batch_size: 64(default) up to 1024
        input_filename: input fasta file, gzipped
        output_filename: desired filename, will be returned as an npz file
    """

    # initialize input parameters
    model = request.args.get("model", "transformer")
    tokenizer = request.args.get("tokenizer", "iupac")
    batch_size = request.args.get("batch_size", "64")
    pretrained_model = request.args.get("pretrained_model", "bert-base")

    input_filename = request.args.get("input_filename")
    output_filename = request.args.get("output_filename")

    try:
        # load file from request
        f = request.files[input_filename]
        f.save(f.filename)

        # decompress file
        os.system("gzip -d {}".format(f.filename))

        # fix filenames
        input_file = f.filename.replace(".gz", "")
        output_file = output_filename + ".npz"

        # embed data
        os.system("tape-embed {} {} {} {} --batch_size {} --tokenizer {}".format(
            model, input_file, output_filename, pretrained_model, batch_size, tokenizer))

        # remove input file
        os.system("rm {}".format(input_file))

        return send_file(output_file, attachment_filename=output_file), 200
    except Exception as e:
        return {"error": "{}".format(e)},400


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
    flask_app.config['UPLOAD_FOLDER'] = "/tape/data/"


    flask_app.add_url_rule(rule="/health/full", view_func=health_check, methods=["GET"])
    flask_app.add_url_rule(rule="/torch/gpu", view_func=get_torch_gpu_settings, methods=["GET"])
    flask_app.add_url_rule(rule='/embed', view_func=embed_data, methods=['POST'])
    return flask_app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8443)
