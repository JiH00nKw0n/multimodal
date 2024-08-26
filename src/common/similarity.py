import os
from typing import List, Dict, Optional, Union
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from PIL import Image
from src.utils.utils import _get_vector_norm


class ImageSimilarityCalculator:
    def __init__(
            self,
            similarity_model_name_or_path: Optional[Union[str, os.PathLike]],
            batch_size: int = 128,
            top_k: int = 3
    ):
        self.similarity_model_name_or_path = similarity_model_name_or_path
        self.batch_size = batch_size
        self.top_k = top_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(self.similarity_model_name_or_path).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.similarity_model_name_or_path)

    def compute_image_similarity(
            self,
            images: List[Image.Image],
            show_progress_bar: Optional[bool] = True
    ) -> Dict[int, List[int]]:
        all_embeddings = []
        for start_index in tqdm(
                range(0, len(images), self.batch_size), desc=f"Encoding {self.batch_size} batches",
                disable=not show_progress_bar):
            batch_images = images[start_index: start_index + self.batch_size]

            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                embeddings = _get_vector_norm(self.model.get_image_features(**inputs))
                embeddings = embeddings.cpu()
                embeddings.to(torch.float32)

            all_embeddings.extend(embeddings)

        all_embeddings = torch.stack(all_embeddings)

        # 유사도 행렬 계산 (각 이미지 쌍 간의 코사인 유사도)
        similarity_matrix = torch.matmul(all_embeddings, all_embeddings.T)

        # 자기 자신과의 유사도를 큰 음수 값으로 설정하여 제외
        diag_indices = torch.arange(similarity_matrix.size(0))
        similarity_matrix[diag_indices, diag_indices] = -float('inf')

        # top_k 유사한 이미지 인덱스를 추출
        top_k_indices = torch.topk(similarity_matrix, self.top_k, dim=1, largest=True).indices

        # 결과를 딕셔너리 형태로 변환
        similarity_dict = {idx: top_k_indices[idx].tolist() for idx in range(similarity_matrix.size(0))}

        return similarity_dict
