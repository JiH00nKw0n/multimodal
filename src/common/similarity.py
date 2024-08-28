import os
from typing import List, Dict, Optional, Union
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from datasets import Dataset
import multiprocessing
import PIL
import requests
from io import BytesIO
from tenacity import retry, stop_after_attempt, wait_fixed


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def load_image(url):
    response = requests.get(url)
    response.raise_for_status()  # HTTP 오류 상태일 경우 예외 발생
    img = PIL.Image.open(BytesIO(response.content)).convert("RGB")
    return img


def process_batch(urls):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        images = pool.map(load_image, urls)
    return [img for img in images if img is not None]


class ImageSimilarityCalculator:
    def __init__(
            self,
            similarity_model_name_or_path: Optional[Union[str, os.PathLike]],
            batch_size: int = 1024,
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
            dataset: Dataset,
            show_progress_bar: Optional[bool] = True
    ) -> Dict[int, List[int]]:
        all_embeddings = []
        for start_index in tqdm(
                range(0, len(dataset), self.batch_size), desc=f"Encoding {self.batch_size} batches",
                disable=not show_progress_bar):
            url_list = dataset[start_index: start_index + self.batch_size]['images']
            batch_images = process_batch(url_list)

            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

            all_embeddings.append(embeddings)  # Use append instead of extend

        # NOTE : Too heavy computation; set device with cpu
        all_embeddings = torch.cat(all_embeddings, dim=0).cpu()  # Concatenate along the first dimension

        # 유사도 행렬 계산 (각 이미지 쌍 간의 코사인 유사도)
        # NOTE: Too heavy computation; Remove cache
        torch.cuda.empty_cache()
        with torch.no_grad():
            similarity_matrix = torch.matmul(all_embeddings, all_embeddings.T)

        # 자기 자신과의 유사도를 큰 음수 값으로 설정하여 제외
        diag_indices = torch.arange(similarity_matrix.size(0))
        similarity_matrix[diag_indices, diag_indices] = -float('inf')

        # top_k 유사한 이미지 인덱스를 추출
        top_k_indices = torch.topk(similarity_matrix, self.top_k, dim=1, largest=True).indices

        # 결과를 딕셔너리 형태로 변환
        similarity_dict = {idx: top_k_indices[idx].tolist() for idx in range(similarity_matrix.size(0))}

        return similarity_dict

    def compute_image_similarity_batched(
            self,
            dataset: Dataset,
            show_progress_bar: Optional[bool] = True
    ) -> Dict[int, List[int]]:
        all_embeddings = []
        similarity_dict = dict()
        for start_index in tqdm(
                range(0, len(dataset), self.batch_size), desc=f"Encoding {self.batch_size} batches",
                disable=not show_progress_bar):
            url_list = dataset[start_index: start_index + self.batch_size]['images']
            batch_images = process_batch(url_list)

            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

            # 유사도 행렬 계산 (각 이미지 쌍 간의 코사인 유사도)
            # TODO: Too heavy computation
            torch.cuda.empty_cache()
            with torch.no_grad():
                similarity_matrix = torch.matmul(embeddings, embeddings.T)

            # 자기 자신과의 유사도를 큰 음수 값으로 설정하여 제외
            diag_indices = torch.arange(similarity_matrix.size(0))
            similarity_matrix[diag_indices, diag_indices] = -float('inf')

            # top_k 유사한 이미지 인덱스를 추출
            top_k_indices = torch.topk(similarity_matrix, self.top_k, dim=1, largest=True).indices

            # 결과를 딕셔너리 형태로 변환
            for idx in range(similarity_matrix.size(0)):
                similarity_dict[idx+start_index] = top_k_indices[idx].tolist()
        return similarity_dict
