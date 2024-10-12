from src.models.configuration_base import BaseConfig
from src.models.modeling_base import BaseModel
from src.models.clipbert import CLIPBertConfig, CLIPBertProcessor, CLIPBertModel
from src.models.clipt5 import CLIPT5Config, CLIPT5Processor, CLIPT5EncoderModel
from src.models.clipinstructor import CLIPInstructorConfig, CLIPInstructorProcessor, CLIPInstructorModel
from src.models.processing_base import BaseProcessor
from src.common.registry import registry
from transformers import CLIPModel, CLIPConfig, CLIPProcessor

__all__ = [
    "BaseConfig",
    "BaseModel",
    "BaseProcessor",
    "CLIPBertConfig",
    "CLIPBertProcessor",
    "CLIPBertModel",
    "CLIPT5Config",
    "CLIPT5Processor",
    "CLIPT5EncoderModel",
    "CLIPInstructorConfig",
    "CLIPInstructorProcessor",
    "CLIPInstructorModel",
    "CLIPModel",
    "CLIPConfig",
    "CLIPProcessor",
]

registry.register_model("CLIPModel")(CLIPModel)
registry.register_model_config("CLIPConfig")(CLIPConfig)
registry.register_processor("CLIPProcessor")(CLIPProcessor)
