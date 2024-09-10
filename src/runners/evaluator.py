import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pytrec_eval
import torch
from datasets import tqdm
from pydantic import Field
from torch.utils.data import DataLoader

from src.collators import BaseCollator
from src.common.registry import registry
from src.runners.base import BaseEvaluator

CollatorType = Type[BaseCollator]


def dummy_collator(batch: List[Dict]) -> List[Dict]:
    """
    A dummy collator function to bypass type errors caused by non-tensor elements in a batch.

    This collator avoids the `TypeError: batch must contain tensors, numbers, dicts or lists;
    found <class 'PIL.Image.Image'>` when handling batches containing non-standard data types
    like PIL images. Instead of performing tensor stacking, it simply returns the batch as-is.

    Args:
        batch (`List[Dict]`): A list of dictionaries, where each dictionary corresponds to an
        element in the batch.

    Returns:
        `List[Dict]`: The input batch is returned unchanged.

    Note:
        For more information on the default PyTorch collator (`default_collate`), you can refer to its
        implementation in the PyTorch repository:
        https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py
    """
    return batch


@registry.register_evaluator("RetrievalEvaluator")
class RetrievalEvaluator(BaseEvaluator):
    """
    An evaluator class for retrieval tasks, which evaluates the model's performance on matching images to text
    and text to images. The input dataset contains image and text embeddings, and the evaluation process computes
    recall metrics at different `k` values.

    Attributes:
        k_values (Optional[List[int]]): A list of `k` values used to compute recall (default is `[1, 5, 10]`).
        qrels_text_to_image (defaultdict): Mapping from text to relevant images for `pytrec_eval`.
        qrels_image_to_text (defaultdict): Mapping from images to relevant texts for `pytrec_eval`.
        image_embeds (List[torch.Tensor]): A list to store image embeddings during evaluation.
        text_embeds (List[torch.Tensor]): A list to store text embeddings during evaluation.
    """

    k_values: Optional[List[int]] = None
    qrels_text_to_image: defaultdict = Field(default_factory=lambda: defaultdict(dict))
    qrels_image_to_text: defaultdict = Field(default_factory=lambda: defaultdict(dict))
    image_embeds: List[torch.Tensor] = Field(default_factory=list)
    text_embeds: List[torch.Tensor] = Field(default_factory=list)

    def _encode_dataset(self, batch_size: int = 128) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Encodes the dataset by processing image and text inputs, generating the necessary embeddings
        for evaluation. The dataset is iterated in batches, and the text and image embeddings are accumulated.

        Args:
            batch_size (int, optional): The number of samples per batch (default is 128).

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                - text_embeds: The text embeddings for the dataset.
                - image_embeds: The image embeddings for the dataset.

        Raises:
            TypeError: If `text` in the dataset is not of type `str` or `list`.
        """

        def process_samples(_samples: List[dict]) -> List[dict]:
            """
            Processes input samples to handle cases where multiple texts correspond to one image.
            If a sample contains multiple texts, it duplicates the sample for each text.

            Args:
                _samples (List[dict]): The input samples, where each sample contains `images` and `text`.

            Returns:
                List[dict]: A list of processed samples, ensuring each sample contains a single text.
            """
            processed_samples = []
            for _sample in _samples:
                _text_sample = _sample['text']
                if isinstance(_text_sample, str):
                    processed_samples.append(_sample)
                elif isinstance(_text_sample, list):
                    _first_sample = _sample.copy()
                    _first_sample['text'] = _text_sample[0]
                    processed_samples.append(_first_sample)

                    for _text in _text_sample[1:]:
                        _other_sample = {'images': None, 'text': _text}
                        processed_samples.append(_other_sample)
                else:
                    raise TypeError(
                        f"Expected `text` to be of type `str` or `list`, but got {type(_text_sample)}"
                    )

            return processed_samples

        image_idx = 0
        text_idx = 0

        # Load dataset using a DataLoader, with the dummy_collator handling input structure
        dataloader = DataLoader(
            self.evaluate_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dummy_collator
        )

        for samples in tqdm(dataloader):
            for sample in samples:
                text_sample = sample['text']
                image_sample = f"image_{image_idx}"

                if isinstance(text_sample, str):
                    # Handle single text for the image
                    text_sample_name = f"text_{text_idx}"
                    self.qrels_text_to_image[text_sample_name][image_sample] = 1
                    self.qrels_image_to_text[image_sample][text_sample_name] = 1
                    text_idx += 1

                elif isinstance(text_sample, list):
                    # Handle multiple texts for the image
                    for _text in text_sample:
                        text_sample_name = f"text_{text_idx}"
                        self.qrels_text_to_image[text_sample_name][image_sample] = 1
                        self.qrels_image_to_text[image_sample][text_sample_name] = 1
                        text_idx += 1

                else:
                    raise TypeError(f"Expected `text` to be of type `str` or `list`, but got {type(text_sample)}")

                image_idx += 1

            # Process the samples
            processed_samples = process_samples(samples)
            inputs = self.data_collator(processed_samples)

            with torch.no_grad():
                outputs = self.model(**inputs)

                _text_embeds = outputs.text_embeds
                _image_embeds = outputs.image_embeds

            self.text_embeds.extend(_text_embeds)
            self.image_embeds.extend(_image_embeds)

        # No need to return the mappings as they are directly stored in qrels
        return self.text_embeds, self.image_embeds

    def evaluate(self, batch_size: Optional[int] = 128):
        """
        Evaluates the retrieval performance using pytrec_eval for metrics calculation.
        Both text-to-image and image-to-text retrieval tasks are evaluated.

        Args:
            batch_size (int, optional): The number of samples per batch (default is 128).

        Returns:
            None: Results are saved to a `result.json` file in the output directory.
        """
        k_values = self.k_values if self.k_values is not None else [1, 5, 10]

        print("Encoding all data...")
        text_embeds, image_embeds = self._encode_dataset(batch_size=batch_size)

        num_text = len(text_embeds)
        num_im = len(image_embeds)

        # Compute similarity matrix between text and image embeddings
        with torch.no_grad():
            dist_matrix = torch.stack(text_embeds) @ torch.stack(image_embeds).T
        dist_matrix = dist_matrix.cpu().detach()

        # Create run dictionaries for pytrec_eval
        run_text_to_image = {}
        run_image_to_text = {}

        # Prepare data for text-to-image retrieval (query: text, docs: images)
        for i in range(num_text):
            run_text_to_image[f"text_{i}"] = {f"image_{j}": float(dist_matrix[i, j].item()) for j in range(num_im)}

        # Prepare data for image-to-text retrieval (query: image, docs: texts)
        dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for ith image
        for i in range(num_im):
            run_image_to_text[f"image_{i}"] = {f"text_{j}": float(dist_matrix[i, j].item()) for j in range(num_text)}

        # Initialize pytrec_eval evaluator for text-to-image retrieval
        evaluator = pytrec_eval.RelevanceEvaluator(self.qrels_text_to_image, {'recall', 'map'})

        # Calculate metrics for text-to-image retrieval
        t2i_results = evaluator.evaluate(run_text_to_image)
        t2i_recalls = {f"Recall@{k}": sum([v[f"recall_{k}"] for v in t2i_results.values()]) / num_text for k in
                       k_values}

        # Initialize pytrec_eval evaluator for image-to-text retrieval
        evaluator = pytrec_eval.RelevanceEvaluator(self.qrels_image_to_text, {'recall', 'map'})

        # Calculate metrics for image-to-text retrieval
        i2t_results = evaluator.evaluate(run_image_to_text)
        i2t_recalls = {f"Recall@{k}": sum([v[f"recall_{k}"] for v in i2t_results.values()]) / num_im for k in k_values}

        # Save the results
        self._save_result({"t2i": t2i_recalls, "i2t": i2t_recalls})


def text_correct(result):
    """
    Checks whether the text is correctly matched to the image by comparing scores in the result dictionary.

    Args:
        result (dict): A dictionary containing scores for text-image pairings.

    Returns:
        bool: True if text matches are correct, False otherwise.
    """
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]


def image_correct(result):
    """
    Checks whether the image is correctly matched to the text by comparing scores in the result dictionary.

    Args:
        result (dict): A dictionary containing scores for text-image pairings.

    Returns:
        bool: True if image matches are correct, False otherwise.
    """
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]


def group_correct(result):
    """
    Checks whether both the image and text are correctly matched in the result.

    Args:
        result (dict): A dictionary containing scores for text-image pairings.

    Returns:
        bool: True if both text and image matches are correct, False otherwise.
    """
    return image_correct(result) and text_correct(result)


@registry.register_evaluator("WinogroundEvaluator")
class WinogroundEvaluator(BaseEvaluator):
    """
    An evaluator class for Winoground dataset. This evaluator computes the matching performance of image-text pairs
    using the model, based on the 'caption_0', 'caption_1', 'image_0', and 'image_1' fields in the dataset.

    It calculates three metrics:
    - Text score: Correctly matches text with images.
    - Image score: Correctly matches images with texts.
    - Group score: Correctly matches both texts and images.

    Attributes:
        winoground_scores (List[Dict[Any, Any]]): A list to store the scores for each example in the dataset.
    """

    winoground_scores: List[Dict[Any, Any]] = Field(default_factory=list)

    def _encode_dataset(self, batch_size: int = 128):
        """
        Encodes the dataset by preparing inputs for different image-text combinations and generates scores
        for each pair.

        Args:
            batch_size (int, optional): The number of samples per batch (default is 128).

        Raises:
            TypeError: If the inputs are not correctly formatted for encoding.
        """
        dataloader = DataLoader(
            self.evaluate_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dummy_collator
        )

        for samples in dataloader:
            # Prepare inputs for each possible combination of text and image
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

            # Collect IDs for each example in the batch
            ids = list(map(lambda d: d['id'], samples))

            with torch.no_grad():
                # Generate output logits for each text-image combination
                output_c0_i0 = self.model(**input_c0_i0)
                output_c1_i0 = self.model(**input_c1_i0)
                output_c0_i1 = self.model(**input_c0_i1)
                output_c1_i1 = self.model(**input_c1_i1)

            # Extract scores for each example and store in winoground_scores
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
        """
        Evaluates the model performance on the Winoground dataset by computing the text, image, and group scores.

        Args:
            batch_size (int, optional): The number of samples per batch (default is 128).

        Outputs:
            Prints the text, image, and group scores. Also saves the results in a 'WINOGROUND.json' file.
        """
        self._encode_dataset(batch_size=batch_size)

        text_correct_count = 0
        image_correct_count = 0
        group_correct_count = 0

        # Compute the number of correct matches for text, image, and group
        for result in self.winoground_scores:
            text_correct_count += 1 if text_correct(result) else 0
            image_correct_count += 1 if image_correct(result) else 0
            group_correct_count += 1 if group_correct(result) else 0

        denominator = len(self.winoground_scores)

        # Print scores
        print("text score:", text_correct_count / denominator)
        print("image score:", image_correct_count / denominator)
        print("group score:", group_correct_count / denominator)

        # Save results to output directory
        self._save_result({
                    "text score": round(text_correct_count / denominator, 3),
                    "image score": round(image_correct_count / denominator, 3),
                    "group score": round(group_correct_count / denominator, 3),
                })


@registry.register_evaluator("SVOEvaluator")
class SVOEvaluator(BaseEvaluator):
    """
    An evaluator class for evaluating Subject-Verb-Object (SVO) datasets. This evaluator compares positive and negative
    image-text pairs based on subjects, verbs, and objects to calculate accuracy.

    Attributes:
        svo_scores (List[Dict[Any, Any]]): A list to store the scores for each SVO example in the dataset.
    """

    svo_scores: List[Dict[Any, Any]] = Field(default_factory=list)

    def _encode_dataset(self, batch_size: int = 128):
        """
        Encodes the dataset by processing image and text inputs for positive and negative samples and
        generates scores for each.

        Args:
            batch_size (int, optional): The number of samples per batch (default is 128).

        Returns:
            None: Stores the scores in `self.svo_scores`.

        Raises:
            TypeError: If inputs are not formatted correctly for encoding.
        """

        def _get_id_list(s: List[Dict]) -> List:
            """
            Extracts the key (`subj_neg`, `verb_neg`, or `obj_neg`) that is True in each sample.

            Args:
                s (List[Dict]): A list of dictionaries, where each dictionary contains the keys `subj_neg`,
                `verb_neg`, and `obj_neg`.

            Returns:
                List: A list of keys that have True values in the respective sample.
            """
            keys_to_check = ['subj_neg', 'verb_neg', 'obj_neg']
            return_list = []

            for sample in s:
                # Find the key that has a True value in the sample
                true_key = next((key for key in keys_to_check if sample.get(key)), None)
                if true_key:
                    return_list.append(true_key)

            return return_list

        dataloader = DataLoader(
            self.evaluate_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dummy_collator
        )

        for samples in dataloader:
            # Prepare inputs for positive and negative image-text pairs
            input_pos = self.data_collator(list(map(lambda d: {
                'text': d['sentences'],
                'images': d['pos_image'],
            }, samples)))

            input_neg = self.data_collator(list(map(lambda d: {
                'text': d['sentences'],
                'images': d['neg_image'],
            }, samples)))

            ids = _get_id_list(s=samples)

            with torch.no_grad():
                # Get model outputs for positive and negative samples
                output_pos = self.model(**input_pos)
                output_neg = self.model(**input_neg)

            # Store scores for each sample
            for idx, example_id in enumerate(ids):
                score_pos = output_pos.logits_per_image[idx].item()
                score_neg = output_neg.logits_per_image[idx].item()

                self.svo_scores.append({
                    "id": example_id,
                    "pos_scores": score_pos > score_neg,
                    "neg_scores": score_neg > score_pos
                })

    def evaluate(self, batch_size: int = 128):
        """
        Evaluates the model on the SVO dataset by calculating positive and negative image-text matching accuracy
        for subjects, verbs, and objects.

        Args:
            batch_size (int, optional): The number of samples per batch (default is 128).

        Returns:
            None: Saves the evaluation results in the output directory.
        """
        self._encode_dataset(batch_size=batch_size)

        def accuracy(samples: List[Dict]) -> Tuple[float, float, float]:
            """
            Calculates the accuracy for positive and negative image-text matches.

            Args:
                samples (List[Dict]): A list of dictionaries containing the scores for each sample.

            Returns:
                Tuple[float, float, float]: The overall accuracy, positive accuracy, and negative accuracy.
            """
            # Calculate negative image accuracy (where `neg_scores` should be False)
            _neg_acc = np.mean([not sample['neg_scores'] for sample in samples])

            # Calculate positive image accuracy (where `pos_scores` should be True)
            _pos_acc = np.mean([sample['pos_scores'] for sample in samples])

            # Calculate overall accuracy (macro accuracy)
            _acc = (_neg_acc + _pos_acc) / 2.0

            return _acc, _pos_acc, _neg_acc

        # Calculate accuracy for subjects
        subj_neg_acc, subj_neg_pos_acc, subj_neg_neg_acc = accuracy(
            list(filter(lambda sample: sample.get('id') == 'subj_neg', self.svo_scores))
        )

        # Calculate accuracy for verbs
        verb_neg_acc, verb_neg_pos_acc, verb_neg_neg_acc = accuracy(
            list(filter(lambda sample: sample.get('id') == 'verb_neg', self.svo_scores))
        )

        # Calculate accuracy for objects
        obj_neg_acc, obj_neg_pos_acc, obj_neg_neg_acc = accuracy(
            list(filter(lambda sample: sample.get('id') == 'obj_neg', self.svo_scores))
        )

        # Calculate overall accuracy
        acc, pos_acc, neg_acc = accuracy(self.svo_scores)

        # Results dictionary
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

        # Save results
        self._save_result(results)


@registry.register_evaluator("AROEvaluator")
class AROEvaluator(BaseEvaluator):
    aro_scores: List[Dict[Any, Any]] = Field(default_factory=list)

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
    crepe_scores: List[Dict[Any, Any]] = Field(default_factory=list)

    def _encode_dataset(self, batch_size: int = 128):
        pass

    def evaluate(self, batch_size: int = 128):
        pass


@registry.register_evaluator("SugarCrepeEvaluator")
class SugarCrepeEvaluator(BaseEvaluator):
    sugar_crepe_scores: List[Dict[Any, Any]] = Field(default_factory=list)

    def _encode_dataset(self, batch_size: int = 128):
        pass

    def evaluate(self, batch_size: int = 128):
        pass


@registry.register_evaluator("VLCEvaluator")
class VLCEvaluator(BaseEvaluator):
    vlc_scores: List[Dict[Any, Any]] = Field(default_factory=list)

    def _encode_dataset(self, batch_size: int = 128):
        pass

    def evaluate(self, batch_size: int = 128):
        pass
