import io
from typing import Any

import numpy as np
from nvflare.apis.utils.decomposers.flare_decomposers import DXODecomposer
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.datum import DatumManager
from nvflare.fuel.utils.fobs.decomposer import Decomposer, Externalizer, Internalizer
from nvflare.fuel.utils.fobs.fobs import register
from nvflare.fuel.utils.fobs.lobs import load_from_file, load_from_bytes

from nvflare.fuel.utils import fobs




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



class CustomDictDecomposer(Decomposer):
    # def __init__(self):
    #     super().__init__()
    #     self.dict_type = dict

    def supported_type(self):
        return dict

    def decompose(self, target: dict, manager: DatumManager = None) -> Any:
        # # need to create a new object; otherwise msgpack will try to decompose this object endlessly.
        # tc = target.copy()
        # manager.register_copy(tc, target)
        # externalizer = Externalizer(manager)
        # return externalizer.externalize(tc)
        return target.tobytes()

    def recompose(self, data: dict, manager: DatumManager = None) -> dict:
        # internalizer = Internalizer(manager)
        # data = internalizer.internalize(data)
        # obj = {} #self.dict_type()
        # for k, v in data.items():
        #     obj[k] = v
        # return obj
        return np.frombuffer(data, dtype=np.float32)

def load_snapshot():
    fobs.register(CustomDictDecomposer)

    # Read the serialized bytes from the file
    file_path = "data"
    with open(file_path, "rb") as f:
        bytes_data = f.read()
    sp = fobs.load_from_bytes(bytes_data)
    print(sp.data.keys())



if __name__ == '__main__':
    # Register the decomposer for DXO
    register(DXODecomposer)
    # Register the decomposer
    fobs.register(NumpyArrayDecomposer)

    # print(fobs.loadf('data'))
    load_snapshot()

