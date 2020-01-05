FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN python --version

RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get update && apt-get install -y git

# For GPU
RUN pip install --upgrade pip
RUN pip install Pillow
RUN pip install numpy
RUN pip install jupyter
RUN pip install matplotlib
RUN pip install PrettyTable
RUN pip install read-roi==1.5.1
RUN pip install opencv-python

WORKDIR /var/project/deeplab/datasets/tooth/init_models
WORKDIR /var/project

COPY . /var/project/

ENTRYPOINT echo "Hello, Tensorflow"

# RUN cd research
# RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim


# FROM tensorflow/tensorflow:1.15.0-py3

# WORKDIR /var/project
# COPY . /var/project

# WORKDIR /var/project

# RUN apt-get install -y git

# RUN pip install Cython
# RUN pip install --requirement requirements.txt
# RUN apt-get install -y libsm6 libxext6 libxrender-dev

# CMD python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"




# CMD python src/tooth.py --data_dir=/var/project/data/ --model_dir=/var/project/logs/ --init_with=coco > logs/output.txt

# sudo docker build --tag=cfil:latest .

# sudo docker run -it --name cfapp --gpus all -v ~/project/logs:/var/project/logs cfil:latest bash
# docker run -it --name cfapp_models --gpus all cfil_models:latest bash

# nohup python src/tooth.py --data_dir=/var/project/data/ --model_dir=/var/project/logs/heads_8/ --init_with=coco > output.txt &
# nohup python /var/project/src/tooth.py --data_dir=/var/project/data/ --model_dir=/var/project/logs/heads_40_decay00001/ --init_with=coco > /var/project/output.txt &
# sudo docker cp src/tooth.py cfapp:/var/project/src/tooth.py