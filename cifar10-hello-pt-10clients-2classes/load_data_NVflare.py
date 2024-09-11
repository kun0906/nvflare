from typing import Any

import numpy as np
from nvflare.apis.utils.decomposers.flare_decomposers import DXODecomposer
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposer import Decomposer
from nvflare.fuel.utils.fobs.fobs import register
from nvflare.fuel.utils.fobs.lobs import load_from_file

# Register the decomposer for DXO
register(DXODecomposer)


class NumpyArrayDecomposer(Decomposer):
    def supported_type(self) -> type:
        return np.ndarray

    def decompose(self, target: np.ndarray) -> Any:
        return target.tobytes()

    def recompose(self, data: Any, manager) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32)


# Register the decomposer
fobs.register(NumpyArrayDecomposer)

# Path to the model_shareables file
model_shareables_path = "../cross_site_val/model_shareables/SRV_server"

# Load the serialized shareable
model_shareable = load_from_file(model_shareables_path)

# Now you can inspect the DXO object
print(model_shareable.data.keys())

# Path to the model_shareables file
result_shareables_path = "../cross_site_val/result_shareables/site-bcm-dgxa100-0003_site-bcm-dgxa100-0013"

# Load the serialized shareable
result_shareable = load_from_file(result_shareables_path)

# Now you can inspect the DXO object
print(result_shareable.data.keys())
