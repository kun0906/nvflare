from typing import Any

import numpy as np
from nvflare.apis.utils.decomposers.flare_decomposers import DXODecomposer
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposer import Decomposer
from nvflare.fuel.utils.fobs.fobs import register
from nvflare.fuel.utils.fobs.lobs import load_from_file


class NumpyArrayDecomposer(Decomposer):
    def supported_type(self) -> type:
        return np.ndarray

    def decompose(self, target: np.ndarray) -> Any:
        return target.tobytes()

    def recompose(self, data: Any, manager) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32)



def load_results(project_dir):
    # Path to the result_shareables file
    result_shareables_path = f"{project_dir}/cross_site_val/result_shareables/site-1_SRV_server"
    # Load the serialized shareable
    result_shareable = load_from_file(result_shareables_path)

    # Now you can inspect the DXO object
    print(result_shareable.data.keys())


def load_model(project_dir):
    # Path to the model_shareables file
    model_shareables_path = f"{project_dir}/cross_site_val/model_shareables/SRV_server"

    # Load the serialized shareable
    model_shareable = load_from_file(model_shareables_path)

    # Now you can inspect the DXO object
    print(model_shareable.data.keys())



if __name__ == '__main__':
    # Register the decomposer for DXO
    register(DXODecomposer)
    # Register the decomposer
    fobs.register(NumpyArrayDecomposer)

    project_dir = '/Users/49751124/cifar10-hello-pt-10clients-2classes/transfer/0998c927-870c-41e4-8e08-f0f578cceb96/workspace'
    load_results(project_dir)

    load_model(project_dir)
