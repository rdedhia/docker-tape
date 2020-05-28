"""
Main entrypoint for the Flask webserver.
"""
from flask import (
    Flask,
    jsonify,
    request,
    send_file,
    render_template
)
import torch
import gzip
import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import json

def gen_df(df, label_list, arrays, lookup_d):
    l = list(arrays.keys())
    labels = []
    for a in l:
        d = arrays[a].item()['avg']
        append_df = pd.DataFrame(d)
        labels.append(lookup_d[a])
        df = df.append(append_df.transpose(), ignore_index=True)
    return df, labels

def visualize_data():

    input_filename = request.args.get("input_filename")
    targets_filename = request.args.get("targets_filename")
    prediction_task = request.args.get("prediction")
    n_components = request.args.get("n_components", 3)
    labels = None
    try: 
        f = request.files[input_filename]
        f.save(f.filename)

        targets_file = request.files[targets_filename]
        targets_file.save(targets_file.filename)
        labels = json.load(open(targets_file.filename))

        print("files recieved")
        os.system("gzip -d {}".format(input_filename))
        input_file = f.filename.replace(".gz", "")
        input_data = np.load(input_file, allow_pickle=True)
        os.system("rm {}".format(input_file))
        print("generating dataframes")
        embed_df, embed_labels = gen_df(pd.DataFrame(), [], input_data, labels)
        print("generating PCA")
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(embed_df)
        principal_df = pd.DataFrame(data = principal_components
                 , columns = ['pc1', 'pc2', 'pc3'])
        principal_df['target'] = embed_labels
        print("generating plot")
        if labels:
            if prediction_task == "classification":
                if n_components == 3:
                    fig = px.scatter_3d(principal_df, x='pc1', y='pc2', z='pc3', color='target', color_discrete_sequence=px.colors.qualitative.G10)
                    fig.write_html('templates/index.html')
                if n_components == 2:
                    fig = px.scatter(principal_df, x='pc1', y='pc2', color='target', color_discrete_sequence=px.colors.qualitative.G10)
                    fig.write_html('templates/index.html')

        else:
            fig = px.scatter_3d(principal_df, x='pc1', y='pc2', z='pc3')
            fig.write_html('templates/index.html')
        return "success", 200
    except Exception as e:
        return {"error": "{}".format(e)}, 400


def load_visualization():
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
    flask_app.add_url_rule(rule='/embed_data', view_func=embed_data, methods=['POST'])
    flask_app.add_url_rule(rule='/visualize_data', view_func=visualize_data, methods=['POST'])
    flask_app.add_url_rule(rule='/show_visualization', view_func=load_visualization, methods=['GET'])
    return flask_app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8443, debug=True)
