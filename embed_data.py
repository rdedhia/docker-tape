import requests
import gzip

tape_url = "http://localhost:8443/embed"

filename = "deeploc_train.fasta.gz"
files = {filename : open(filename, 'rb')}
params = {"input_filename": filename, 
		  "output_filename": "deeploc_transformer", 
		  "model": "transformer",
		  "pretrained_model": "bert-base"}
r = requests.post(tape_url, files=files, params=params)
print(r.text)