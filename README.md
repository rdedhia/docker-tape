# docker-tape
Docker deployment for Flask app of songlab-cal TAPE repo for protein embedding analysis

To build the image, run `make build`, or:

```
docker build -t tape .
```

To run the container, run `make run-cpu` or `make run-gpu` depending on which device you want to use:

CPU:

```
docker run -p 8443:8443 tape
```

GPU:

```
docker run --runtime=nvidia -p 8443:8443 tape
```

Test the running application with `CURL`. There is a `torch/gpu` route which specifies information about the GPU. To
test, run `make test` or use:

```
curl http://0.0.0.0:8443/torch/gpu
```

The response on a CPU:

```
{
    "cuda": false
}
```

The response on a `p3.8xlarge` instance with 4 `Tesla V100` GPUs:

```
{
    "cuda":true,
    "device_count":4,
    "device_name":"Tesla V100-SXM2-16GB"
}
```
