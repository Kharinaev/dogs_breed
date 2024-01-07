from dataclasses import dataclass


@dataclass
class ClientRunConfig:
    image_path: str
    class_num_dict_path: str


@dataclass
class ClientTestsConfig:
    do_tests: bool
    image_path: str
    on_gpu: bool
    local_model_path: str


@dataclass
class Params:
    client_run: ClientRunConfig
    client_test: ClientTestsConfig
