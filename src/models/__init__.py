from src.models.configuration_base import *
from src.models.modeling_base import *
from src.models.fusemix import *
from src.models.processing_base import BaseProcessor
from src.common.registry import registry

__all__ = [
    "BaseConfig",
    "BaseModel",
    "BaseProcessor",
    "FuseMixConfig",
    "FuseMixModel",
    "FuseMixProcessor",
]

# registry.register_model("CLIPModel")(CLIPModel)
# registry.register_model_config("CLIPConfig")(CLIPConfig)

