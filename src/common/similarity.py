import os
from typing import List, Dict, Optional, Union
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from datasets import Dataset
from src.utils.utils import process_batch


class ImageSimilarityCalculator:
    """
    A class for calculating image similarity using a pretrained model. This class provides methods to compute
    image similarity scores for a batch of images and return the top-k most similar images.

    Args:
        similarity_model_name_or_path (`str` or `os.PathLike`, *optional*):
            The path or name of the pretrained model to be used for similarity calculation.
        batch_size (`int`, *optional*, defaults to 128):
            The number of images to process in each batch.
        top_k (`int`, *optional*, defaults to 3):
            The number of top-k similar images to return for each image.

    Attributes:
        similarity_model_name_or_path (`str` or `os.PathLike`):
            Path or name of the model used for similarity calculations.
        batch_size (`int`):
            The batch size for processing images.
        top_k (`int`):
            The number of top-k similar images to return.
        device (`torch.device`):
            The device (CPU or GPU) to run the computations on.
        model (`transformers.PreTrainedModel`):
            The pretrained model used to compute image embeddings.
        processor (`transformers.PreTrainedProcessor`):
            The processor used to preprocess images.
    """

    def __init__(
        self,
        similarity_model_name_or_path: Optional[Union[str, os.PathLike]],
        batch_size: int = 128,
        top_k: int = 3
    ):
        """
        Initializes the ImageSimilarityCalculator with the specified model and batch size.

        Args:
            similarity_model_name_or_path (`str` or `os.PathLike`):
                Path or model name of the pretrained model to be used.
            batch_size (`int`, *optional*, defaults to 128):
                The number of images to process in a batch.
            top_k (`int`, *optional*, defaults to 3):
                The number of top-k similar images to return for each image.
        """
        self.similarity_model_name_or_path = similarity_model_name_or_path
        self.batch_size = batch_size
        self.top_k = top_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and processor
        self.model = AutoModel.from_pretrained(self.similarity_model_name_or_path).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.similarity_model_name_or_path)

        # NOTE: Add for caching
        self.emb_dir = f'{os.getenv("DATA_ROOT_DIR")}/{os.getenv("TARGET_DATASET")}/embeddings'


    def compute_image_similarity(
        self,
        dataset: Dataset,
        show_progress_bar: Optional[bool] = True
    ) -> Dict[int, List[int]]:
        """
        Computes image similarity for all images in the dataset.

        Args:
            dataset (`Dataset`):
                The dataset containing images to compare.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Whether to display a progress bar during the similarity computation.

        Returns:
            `Dict[int, List[int]]`:
                A dictionary where keys are image indices and values are lists of top-k most similar image indices.
        """
        cache_list = list(map(lambda x: x.split('/')[-1], glob(f'{self.emb_dir}/*.pt')))

        all_embeddings = []
        for start_index in tqdm(
            range(0, len(dataset), self.batch_size),
            desc=f"Encoding {self.batch_size} batches",
            disable=not show_progress_bar
        ):
            cache_fname =f'bs_{start_index}.pt'

            if cache_fname in cache_list:
                embeddings = torch.load(embeddings, f'{self.emb_dir}/{cache_fname}')
            else:
                url_list = dataset[start_index: start_index + self.batch_size]['images']
                batch_images = process_batch(url_list)

                inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)

                # Compute image embeddings
                with torch.no_grad():
                    embeddings = self.model.get_image_features(**inputs)
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                
                torch.save(embeddings, f'{self.emb_dir}/{cache_fname}')

            all_embeddings.append(embeddings.cpu())

        # Concatenate all embeddings and move to CPU
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Compute cosine similarity
        torch.cuda.empty_cache()
        similarity_dict = {}
        for start_index in tqdm(
                range(0, len(dataset), self.batch_size), desc=f"Mining {self.batch_size} batches"):
            emb = all_embeddings[start_index: start_index + self.batch_size]                
            with torch.no_grad():
                similarity_matrix = torch.matmul(emb, all_embeddings.T)
            
            # TODO; not working for batched similarity matrix computation
            diag_indices = torch.arange(similarity_matrix.size(0))
            similarity_matrix[diag_indices, diag_indices] = -float('inf')

            # Manually retrieve one more and remove it, which is identical to itself.
            # top_k 유사한 이미지 인덱스를 추출
            top_k_indices = torch.topk(similarity_matrix, self.top_k+1, dim=1, largest=True).indices

            # 결과를 딕셔너리 형태로 변환
            for idx in range(similarity_matrix.size(0)):
                similarity_dict[start_index + idx] = top_k_indices[idx].tolist()[1:]

        # with torch.no_grad():
        #     similarity_matrix = torch.matmul(all_embeddings, all_embeddings.T)

        # # Set self-similarity to negative infinity to exclude
        # diag_indices = torch.arange(similarity_matrix.size(0))
        # similarity_matrix[diag_indices, diag_indices] = -float('inf')

        # # Extract top-k similar images for each image
        # top_k_indices = torch.topk(similarity_matrix, self.top_k, dim=1, largest=True).indices

        # # Convert to dictionary format
        # similarity_dict = {idx: top_k_indices[idx].tolist() for idx in range(similarity_matrix.size(0))}

        return similarity_dict
