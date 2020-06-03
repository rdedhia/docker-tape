"""Script to test embedding function of local Flask webserver"""
import requests


tape_url = "http://localhost:8443/embed"
filename = "deeploc_train.fasta.gz"
params = {
    "input_filename": filename,
    "output_filename": "deeploc_transformer",
    "model": "transformer",
    "pretrained_model": "bert-base",
}
with open(filename, "rb") as handle:
    response = requests.post(tape_url, files=handle, params=params)
    print(response.text)
