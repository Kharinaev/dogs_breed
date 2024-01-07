# DL toolkit

MLops course project with demonstration case

## Dogs breed classification

Neural network classification for dogs breed based on their images

![dataset_image](https://production-media.paperswithcode.com/datasets/Stanford_Dogs-0000000577-91cb15b5_1ABtNf7.jpg)

### Data

The subset of breeds from [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) dataset is used (64/120 breeds, ~173 examples avg. for each breed)

### Model

The [ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) model from torchvision is used

### Usage

#### Prerequisites

- [Poetry](https://python-poetry.org/)
    - For package management
- [Docker](https://www.docker.com/)
    - For running Nvidia Triton Server

#### Installation

To install, run the following commands
```
git clone https://github.com/Kharinaev/dltoolkit.git
cd dltoolkit
```

Install project dependencies (you may create a venv for project)

```
poetry install
```

#### Model interface

There is a file `commands.py` in the root directory that is the entry point to the project interface

In file `./configs/config.yaml` you can find configs for dataset, model, train and inference

To train model

```
python commands.py train
```

To inference model

```
python commands.py infer
```

To run MLFlow server with model

```
python commands.py run_server
```

#### Nvidia Triton Server

You can deploy Nvidia Triton Server with current model using Docker

```
docker-compose -f triton_server/docker-compose.yaml up --build
```

After that you can `client.py` to run simple tests for deployed model and get predictions on image (image and tests configures in file `./configs/client_config.yaml`, you can use images from `./tests/images`)

```
python commands.py triton_client
```
