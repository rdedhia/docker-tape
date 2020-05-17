build:
	docker build -t tape .

run-cpu:
	docker run -p 8443:8443 tape

run-gpu:
	docker run --runtime=nvidia -p 8443:8443 tape

test:
    curl http://0.0.0.0:8443/torch/gpu
