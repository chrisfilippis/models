docker stop cfil_models

docker rm cfil_models

docker build --tag=cfil_models:latest .

docker run -it --name cfil_models --gpus all cfil_models:latest bash