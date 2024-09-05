import json
import os
from typing import List, Optional, Any, Union, Tuple, Dict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from pydantic import BaseModel, ConfigDict, Field
from transformers import PreTrainedModel
from src.common.collator import SequenceTextCollator
from datasets import Dataset, tqdm
from src.common.registry import registry


def _pad_and_convert_to_tensor(data: List[List[int]], max_length: int) -> list[list[int]]:
    padded_data = [lst + [-1] * (max_length - len(lst)) for lst in data]

    return padded_data


class BaseEvaluator(BaseModel):
    model: PreTrainedModel
    data_collator: SequenceTextCollator
    evaluate_dataset: Dataset
    output_dir: Optional[Union[str, os.PathLike]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.model.eval()

    def _encode_dataset(self, batch_size: int):
        raise NotImplementedError

    def evaluate(self, batch_size: int):
        raise NotImplementedError


@registry.register_evaluator("RetrievalEvaluator")
class RetrievalEvaluator(BaseEvaluator):
    k_values: Optional[List[int]] = None
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
            t2i_recalls[f"Recall@{k_val}"] = round(t2i, 2)
            i2t_recalls[f"Recall@{k_val}"] = round(i2t, 2)

        os.makedirs(self.output_dir, exist_ok=True)

        if self.output_dir is not None:
            with open(os.path.join(self.output_dir, 'result.json'), "w") as f:
                json.dump({"t2i": t2i_recalls, "i2t": i2t_recalls}, f, indent=2)


def text_correct(result):
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]


def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]


def group_correct(result):
    return image_correct(result) and text_correct(result)


@registry.register_evaluator("WinogroundEvaluator")
class WinogroundEvaluator(BaseEvaluator):
    # image_to_text_map[i] gives the corresponding text indices for the ith image
    # (as there are multiple pieces of text for each image)
    winoground_scores: List[Dict[Any]] = Field(default_factory=list)

    def _encode_dataset(self, batch_size: int = 128):
        dataloader = DataLoader(self.evaluate_dataset, batch_size=batch_size, shuffle=False)

        for samples in dataloader:
            input_c0_i0 = self.data_collator(list(map(lambda d: {
                'text': d['caption_0'],
                'images': d['image_0'],
            }, samples)))
            input_c1_i0 = self.data_collator(list(map(lambda d: {
                'text': d['caption_1'],
                'images': d['image_0'],
            }, samples)))
            input_c0_i1 = self.data_collator(list(map(lambda d: {
                'text': d['caption_0'],
                'images': d['image_1'],
            }, samples)))
            input_c1_i1 = self.data_collator(list(map(lambda d: {
                'text': d['caption_1'],
                'images': d['image_1'],
            }, samples)))

            # 모든 샘플에 대해 ID를 수집
            ids = list(map(lambda d: d['id'], samples))

            with torch.no_grad():
                output_c0_i0 = self.model(**input_c0_i0)
                output_c1_i0 = self.model(**input_c1_i0)
                output_c0_i1 = self.model(**input_c0_i1)
                output_c1_i1 = self.model(**input_c1_i1)

            # 각 배치 내의 각 예제에 대해 개별적으로 점수를 추출
            for idx, example_id in enumerate(ids):
                score_c0_i0 = output_c0_i0.logits_per_image[idx].item()
                score_c1_i0 = output_c1_i0.logits_per_image[idx].item()
                score_c0_i1 = output_c0_i1.logits_per_image[idx].item()
                score_c1_i1 = output_c1_i1.logits_per_image[idx].item()

                self.winoground_scores.append({
                    "id": example_id,
                    "c0_i0": score_c0_i0,
                    "c0_i1": score_c0_i1,
                    "c1_i0": score_c1_i0,
                    "c1_i1": score_c1_i1
                })

    def evaluate(self, batch_size: int = 128):

        self._encode_dataset(batch_size=batch_size)

        text_correct_count = 0
        image_correct_count = 0
        group_correct_count = 0
        for result in self.winoground_scores:
            text_correct_count += 1 if text_correct(result) else 0
            image_correct_count += 1 if image_correct(result) else 0
            group_correct_count += 1 if group_correct(result) else 0

        denominator = len(self.winoground_scores)

        print("text score:", text_correct_count / denominator)
        print("image score:", image_correct_count / denominator)
        print("group score:", group_correct_count / denominator)

        os.makedirs(self.output_dir, exist_ok=True)

        if self.output_dir is not None:
            with open(os.path.join(self.output_dir, 'WINOGROUND.json'), "w") as f:
                json.dump({
                    "text score": round(text_correct_count / denominator, 3),
                    "image score": round(image_correct_count / denominator, 3),
                    "group score": round(group_correct_count / denominator, 3),
                }, f, indent=2)


@registry.register_evaluator("SVOEvaluator")
class SVOEvaluator(BaseEvaluator):
    # image_to_text_map[i] gives the corresponding text indices for the ith image
    # (as there are multiple pieces of text for each image)
    svo_scores: List[Dict[Any]] = Field(default_factory=list)

    def _encode_dataset(self, batch_size: int = 128):
        def _get_id_list(s: List[Dict]) -> List:
            # 검사할 key 목록
            keys_to_check = ['subj_neg', 'verb_neg', 'obj_neg']
            return_list = []
            # dict_list의 각 dict에 대해 처리
            for sample in s:
                # 세 가지 key 중 True 값을 가진 key를 찾습니다.
                true_key = next((key for key in keys_to_check if sample.get(key)), None)

                # True 값을 가진 key의 이름을 'id'라는 key의 값으로 추가합니다.
                if true_key:
                    return_list.append(true_key)

            return return_list

        dataloader = DataLoader(self.evaluate_dataset, batch_size=batch_size, shuffle=False)

        for samples in dataloader:
            input_pos = self.data_collator(list(map(lambda d: {
                'text': d['sentences'],
                'images': d['pos_url'],
            }, samples)))
            input_neg = self.data_collator(list(map(lambda d: {
                'text': d['sentences'],
                'images': d['neg_url'],
            }, samples)))

            ids = _get_id_list(s=samples)

            with torch.no_grad():
                output_pos = self.model(**input_pos)
                output_neg = self.model(**input_neg)

            for idx, example_id in enumerate(ids):
                score_pos = output_pos.logits_per_image[idx].item()
                score_neg = output_neg.logits_per_image[idx].item()

                self.svo_scores.append({
                    "id": example_id,
                    "pos_scores": score_pos > score_neg,
                    "neg_scores": score_neg > score_pos
                })

    def evaluate(self, batch_size: int = 128):
        self._encode_dataset(batch_size=batch_size)

        def accuracy(samples: List[Dict]):
            # 부정 이미지의 정확도 계산 (neg_scores가 False인 경우)
            _neg_acc = np.mean([not sample['neg_scores'] for sample in samples])

            # 긍정 이미지의 정확도 계산 (pos_scores가 True인 경우)
            _pos_acc = np.mean([sample['pos_scores'] for sample in samples])

            # 매크로 정확도 계산
            _acc = (neg_acc + pos_acc) / 2.0

            return _acc, _pos_acc, _neg_acc

        subj_neg_acc, subj_neg_pos_acc, subj_neg_neg_acc = accuracy(
            list(filter(lambda sample: sample.get('id') == 'subj_neg', self.svo_scores))
        )
        verb_neg_acc, verb_neg_pos_acc, verb_neg_neg_acc = accuracy(
            list(filter(lambda sample: sample.get('id') == 'verb_neg', self.svo_scores))
        )
        obj_neg_acc, obj_neg_pos_acc, obj_neg_neg_acc = accuracy(
            list(filter(lambda sample: sample.get('id') == 'obj_neg', self.svo_scores))
        )
        acc, pos_acc, neg_acc = accuracy(self.svo_scores)

        results = {
            "ALL": {
                "Avg Accuracy": acc,
                "Pos Accuracy": pos_acc,
                "Neg Accuracy": neg_acc,
            },
            "Subj": {
                "Avg Accuracy": subj_neg_acc,
                "Pos Accuracy": subj_neg_pos_acc,
                "Neg Accuracy": subj_neg_neg_acc,
            },
            "Verb": {
                "Avg Accuracy": verb_neg_acc,
                "Pos Accuracy": verb_neg_pos_acc,
                "Neg Accuracy": verb_neg_neg_acc,
            },
            "Obj": {
                "Avg Accuracy": obj_neg_acc,
                "Pos Accuracy": obj_neg_pos_acc,
                "Neg Accuracy": obj_neg_neg_acc,
            }
        }

        os.makedirs(self.output_dir, exist_ok=True)

        if self.output_dir is not None:
            with open(os.path.join(self.output_dir, 'SVO.json'), "w") as f:
                json.dump(results, f, indent=2)


@registry.register_evaluator("AROEvaluator")
class AROEvaluator(BaseEvaluator):
    aro_scores: List[Dict[Any]] = Field(default_factory=list)

    def _encode_dataset(self, batch_size: int = 128):
        pass

    def evaluate(self, batch_size: int = 128):
        """
            Scores: N x 1 x 2, i.e. first caption is the perturbed one, second is the positive one
        """
        self._encode_dataset(batch_size=batch_size)

        # if isinstance(scores, tuple):
        #     scores_i2t = scores[1]
        #     scores_t2i = scores[0]
        # else:
        #     scores_t2i = scores
        #     scores_i2t = scores
        #
        # preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        # correct_mask = (preds == 1)
        # result_records = []
        # all_attributes = np.array(self.all_attributes)
        # for attr in np.unique(all_attributes):
        #     attr_mask = (all_attributes == attr)
        #     if attr_mask.sum() < 25:
        #         continue
        #     result_records.append({
        #         "Attributes": attr,
        #         "Accuracy": correct_mask[attr_mask].mean(),
        #         "Count": attr_mask.sum(),
        #         "Dataset": "Visual Genome Attribution"
        #     })
        # return result_records


@registry.register_evaluator("CrepeEvaluator")
class CrepeEvaluator(BaseEvaluator):
    crepe_scores: List[Dict[Any]] = Field(default_factory=list)

    def _encode_dataset(self, batch_size: int = 128):
        pass

    def evaluate(self, batch_size: int = 128):
        pass


@registry.register_evaluator("SugarCrepeEvaluator")
class SugarCrepeEvaluator(BaseEvaluator):
    sugar_crepe_scores: List[Dict[Any]] = Field(default_factory=list)

    def _encode_dataset(self, batch_size: int = 128):
        pass

    def evaluate(self, batch_size: int = 128):
        pass


@registry.register_evaluator("VLCEvaluator")
class VLCEvaluator(BaseEvaluator):
    vlc_scores: List[Dict[Any]] = Field(default_factory=list)

    def _encode_dataset(self, batch_size: int = 128):
        pass

    def evaluate(self, batch_size: int = 128):
        pass