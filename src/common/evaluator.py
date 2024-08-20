import json
import os
from typing import List, Optional, Any, Union, Tuple
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from pydantic import BaseModel, ConfigDict, Field
from transformers import PreTrainedModel
from src.common.collator import SequenceTextCollator
from datasets import Dataset
from .registry import registry


def _pad_and_convert_to_tensor(data: List[List[int]], max_length: int) -> list[list[int]]:
    padded_data = [lst + [-1] * (max_length - len(lst)) for lst in data]

    return padded_data


class BaseEvaluator(BaseModel):
    model: PreTrainedModel
    data_collator: SequenceTextCollator
    evaluate_dataset: Dataset

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.model.eval()

    def _encode_dataset(self):
        raise NotImplementedError

    def evaluate(self, k_values: Optional[List[int]]):
        raise NotImplementedError


@registry.register_evaluator("RetrievalEvaluator")
class RetrievalEvaluator(BaseEvaluator):
    k_values: Optional[List[int]] = None
    output_dir: Optional[Union[str, os.PathLike]] = None
    # image_to_text_map[i] gives the corresponding text indices for the ith image
    # (as there are multiple pieces of text for each image)
    image_to_text_map: List[List[int]] = Field(default_factory=list)

    # text_to_image_map[i] gives the corresponding image index for the ith text
    text_to_image_map: List[List[int]] = Field(default_factory=list)

    image_embeds: List[torch.Tensor] = Field(default_factory=list)
    text_embeds: List[torch.Tensor] = Field(default_factory=list)

    def _encode_dataset(self, batch_size: int = 128) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        image_idx = 0
        text_idx = 0
        max_captions_per_image = 1
        dataloader = DataLoader(self.evaluate_dataset, batch_size=batch_size, shuffle=False)
        for samples in dataloader:

            for sample in samples:
                text_sample = sample['text']
                if isinstance(text_sample, str):
                    captions_per_image = 1
                elif isinstance(text_sample, list):
                    captions_per_image = len(text_sample)
                    if captions_per_image > max_captions_per_image:
                        max_captions_per_image = captions_per_image
                else:
                    raise TypeError()
                text_indices = list(range(text_idx, text_idx + captions_per_image))
                self.image_to_text_map.append(text_indices)
                text_idx += captions_per_image
                self.text_to_image_map += [image_idx] * captions_per_image
                image_idx += 1

            inputs = self.data_collator(samples)
            with torch.no_grad():
                outputs = self.model(**inputs)

                _text_embeds = outputs.text_embeds
                _image_embeds = outputs.image_embeds

            self.text_embeds.extend(_text_embeds)
            self.image_embeds.extend(_image_embeds)

        text_embeds = torch.stack(self.text_embeds)
        image_embeds = torch.stack(self.image_embeds)

        text_to_image_map = torch.LongTensor(
            self.text_to_image_map
        ).to(text_embeds.device)
        image_to_text_map = torch.LongTensor(
            _pad_and_convert_to_tensor(self.image_to_text_map, max_length=max_captions_per_image)
        ).to(text_embeds.device)

        return text_embeds, image_embeds, text_to_image_map, image_to_text_map

    def evaluate(self, batch_size: Optional[int] = 128):
        k_values = self.k_values if self.k_values is not None else [1, 5, 10]

        print("Encoding all data...")
        text_embeds, image_embeds, text_to_image_map, image_to_text_map = self._encode_dataset(batch_size=batch_size)

        num_text = text_embeds.shape[0]
        num_im = image_embeds.shape[0]
        captions_per_image = image_to_text_map.shape[1]

        dist_matrix = text_embeds @ image_embeds.T  # dist_matrix[i] gives logits for ith text
        dist_matrix = dist_matrix.cpu()

        # sort in descending order; first is the biggest logit
        indices = torch.argsort(dist_matrix, dim=1, descending=True)
        indices = indices.to(text_embeds.device)

        text_to_image_recall = []

        for k in k_values:
            # extract top k indices only
            top_k = indices[:, :k]

            # correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
            correct = torch.eq(top_k, text_to_image_map.unsqueeze(-1)).any(dim=1)

            num_correct = correct.sum().item()
            text_to_image_recall.append(num_correct / num_text)

        dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

        # sort in descending order; first is the biggest logit
        indices = torch.argsort(dist_matrix, dim=1, descending=True)
        indices = indices.to(text_embeds.device)

        image_to_text_recall = []

        for k in k_values:
            # extract top k indices only
            top_k = indices[:, :k]

            correct = torch.zeros((num_im,), dtype=torch.bool).cuda()

            # for each image, check whether one of the 5 relevant captions was retrieved
            # check if image matches its ith caption (for i=0..4)
            for i in range(captions_per_image):
                contains_index = torch.eq(top_k, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
                correct = torch.logical_or(correct, contains_index)

            num_correct = correct.sum().item()
            image_to_text_recall.append(num_correct / num_im)  #

        t2i_recalls, i2t_recalls = {}, {}

        for k_val, t2i, i2t in zip(k_values, text_to_image_recall, image_to_text_recall):
            t2i_recalls[f"Recall@{k}"] = round(t2i, 2)
            i2t_recalls[f"Recall@{k}"] = round(i2t, 2)

        if self.output_dir is not None:
            with open(self.output_dir, "w") as f:
                json.dump({"t2i": t2i_recalls, "i2t": i2t_recalls}, f, indent=2)
