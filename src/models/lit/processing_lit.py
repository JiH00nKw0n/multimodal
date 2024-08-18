from src.models.processing_base import BaseProcessor

IMG_PROCESSOR_CLASS = ("EfficientNetImageProcessor", "BitImageProcessor")
TOKENIZER_CLASS = ("XLMRobertaTokenizer", "XLMRobertaTokenizerFast")


class LiTProcessor(BaseProcessor):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = IMG_PROCESSOR_CLASS
    tokenizer_class = TOKENIZER_CLASS

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)