# Docker TAPE

This repository provides a GPU-enabled Docker deployment for a Flask app around the 
[Tasks Assessing Protein Embeddings](https://github.com/songlab-cal/tape) (TAPE) project. Maintained by researchers
at UC Berkeley, TAPE provides models for a variety of protein prediction tasks.

The Docker/Flask application provides two main capabilities:
* Generating embeddings from protein sequences using one of TAPE's pretrained models by wrapping the `tape-embed`
method of the TAPE cli. These embeddings can be used as features for downstream prediction tasks. Embedding sequences
is very compute intensive, so can be accelerated by using a GPU. For this reason, the `Dockerfile` relies on an
`Nvidia` base image and is GPU compatible. Furthermore, we include instructions for setting up an EC2 instance with
GPUs in Amazon Web Services (AWS) with `AWS Cloudformation` to run the Docker container.
* Visualizing the embeddings and labels for a given dataset with principal component analysis in 2 or 3 dimensions,
through an interact `plotly` plot.

Along with the server, the application contains a simple user interface for leveraging the two capabilities. 

## Running Locally

### Flask (no Docker)

If you would like to run the Flask application outside of Docker, set up a `Python 3.6` virtual environment, and run
the following from the top level of the project:

```
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Then, start the application:

```
python tape/server.py
```

You can access the user interface in your browser at `http://localhost:8443/`

### Local Docker

Build the image:

```
docker build -t tape .
```

Run the container, with a different runtime depending on which device you want to use. **Note:** Trying to run the
container with `--runtime=nvidia` will cause an error if there are no GPUs available on your machine:

CPU:

```
docker run -p 443:8443 tape
```

GPU:

```
docker run --runtime=nvidia -p 443:8443 tape
```

Access the user interface in your browser at `http://localhost:443/`

## Running on EC2

### EC2 Deployment

AWS account permissions

Keypair

Cloudformation

SSH

Nvidia

Docker build/run

Grab hostname and run app

Restart instance w/ possible Nvidia error

### Docker Deployment on EC2

If you would like to see information about the GPU(s) deployed on your instance, use the `torch/gpu` route:

```
curl http://<EC2_PUBLIC_HOSTNAME>/torch/gpu
```

The response on a `p3.8xlarge` instance with 4 `Tesla V100` GPUs:

```
{
    "cuda":true,
    "device_count":4,
    "device_name":"Tesla V100-SXM2-16GB"
}
```

## Embedding Sequences

Slow for large datasets, especially if using CPU. Can use the test file in `tests/test-files/deeploc_small.fasta` for
a quick test.

## PCA Visualization

Can use data files from our [fork](https://github.com/rdedhia/tape) of TAPE. The data files are located in the `data`
directory of the project, and are described in detail
in [Notebooks.md](https://github.com/rdedhia/tape/blob/master/Notebooks.md).
