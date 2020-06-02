"""
Main entrypoint for the Flask webserver.
"""
from flask import (
    Flask,
    jsonify,
    request,
    send_file,
    render_template,
    redirect,
    url_for
)
import torch
import gzip
import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import json



def gen_arr(arrays, lookup_d):
    keys = arrays.files
    output, labels = [], []
    for key in keys:
        d = arrays[key].item()['avg']
        labels.append(lookup_d[key])
        output.append(d)
    return np.array(output), labels


def upload():
    return render_template("file_upload_form.html")

def visualize_data():

    # load n_components
    n_components = int(request.form.get("n_components"))

    # load input data file and save to disk
    f = request.files.get("input_data")
    f.save(f.filename)

    # load targets file and save to disk
    targets_file = request.files.get("targets")
    targets_file.save(targets_file.filename)

    # create lookup dictionary for targets
    lookup_d = json.load(open(targets_file.filename))

    print("files recieved")
    os.system("gzip -d {}".format(f.filename))
    input_file = f.filename.replace(".gz", "")
    input_data = np.load(input_file, allow_pickle=True)

    # cleanup saved files
    os.system("rm {}".format(input_file))
    os.system("rm {}".format(targets_file.filename))

    print("generating dataframes")
    embed_arr, embed_labels = gen_arr(input_data, lookup_d)
    print("generating PCA")
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(embed_arr)
    principal_df = pd.DataFrame(data = principal_components
             , columns = ['pc1', 'pc2', 'pc3'])
    principal_df['target'] = embed_labels
    print("generating plot")
    output_html = '{}.html'.format(input_file)
    if n_components == 3:
        fig = px.scatter_3d(principal_df, x='pc1', y='pc2', z='pc3', color='target', color_discrete_sequence=px.colors.qualitative.G10)
    if n_components == 2:
        fig = px.scatter(principal_df, x='pc1', y='pc2', color='target', color_discrete_sequence=px.colors.qualitative.G10)

    fig.write_html("templates/index.html")

    return redirect(url_for('show_visualization'))

def show_visualization():
    return render_template("index.html")
    
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
    model = request.form.get("model", "transformer")
    tokenizer = request.form.get("tokenizer", "iupac")
    batch_size = int(request.form.get("batch_size", "64"))
    pretrained_model = request.form.get("pretrained_model", "bert-base")
    output_filename = request.form.get("output_filename")

    # try:
        # load file from request
    f = request.files.get("input_filename")
    f.save(f.filename)

    # decompress file
    os.system("gzip -d {}".format(f.filename))

    # fix filenames
    input_file = f.filename.replace(".gz", "")
    output_file = output_filename + ".npz"

    # embed data
    os.system("tape-embed {} {} {} {} --batch_size {} --tokenizer {}".format(
        model, input_file, output_filename, pretrained_model, batch_size, tokenizer))

    # remove input file and move output file into output directory
    os.system("rm {}".format(input_file))
    os.system("mv {} output_data".format(output_file))

    return redirect(url_for('download_npz', output_filename=output_file))
    # except Exception as e:
    #     return {"error": "{}".format(e)},400

def download_npz(output_filename):
    return render_template("success.html", output_filename=output_filename)


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
    flask_app = Flask(__name__, template_folder='templates')
    flask_app.config['UPLOAD_FOLDER'] = "/tape/output_data/"


    flask_app.add_url_rule(rule="/health/full", view_func=health_check, methods=["GET"])
    flask_app.add_url_rule(rule="/torch/gpu", view_func=get_torch_gpu_settings, methods=["GET"])
    flask_app.add_url_rule(rule='/embed_data', view_func=embed_data, methods=['POST'])
    flask_app.add_url_rule(rule='/generate_visualization', view_func=visualize_data, methods=['POST'])
    flask_app.add_url_rule(rule='/', view_func=upload)
    flask_app.add_url_rule(rule='/show_visualization', view_func=show_visualization)
    flask_app.add_url_rule(rule='/<output_filename>/download', view_func=download_npz)
    return flask_app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8443, debug=True)
