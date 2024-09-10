from typing import Optional

from datasets import load_dataset, DatasetDict

from src.common import registry
from src.datasets.builder import HardSequenceTextDatasetWithImageBuilder


@registry.register_builder('ARODatasetBuilder')
class ARODatasetBuilder(HardSequenceTextDatasetWithImageBuilder):
    """
    A builder class for creating the ARO (Attribute, Relation, and Order) dataset.
    It extends `HardSequenceTextDatasetWithImageBuilder` and combines multiple datasets related to attributes, relations,
    and order understanding tasks.

    Attributes:
        split (Optional[str]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'aro').
    """
    split: Optional[str] = 'train'
    name: Optional[str] = 'aro'

    def build_dataset(self) -> DatasetDict:
        """
        Builds and returns the ARO dataset as a `DatasetDict`. This dataset combines multiple datasets including
        Visual Genome attributes, relations, and Flickr/COCO order datasets.

        Returns:
            DatasetDict: The combined dataset containing attributes, relations, and order datasets.
        """
        # Load and preprocess VG Attribute dataset
        vg_attribute_dataset = load_dataset(
            "yjkimstats/ARO_VG_Attribute_fmt", trust_remote_code=True, split=self.split
        )
        vg_attribute_dataset = vg_attribute_dataset.map(
            lambda x: {'text': [x['text']]}
        )

        # Load and preprocess VG Relation dataset
        vg_relation_dataset = load_dataset(
            "yjkimstats/ARO_VG_Relation", trust_remote_code=True, split=self.split
        )
        vg_relation_dataset = vg_relation_dataset.map(
            lambda x: {'text': [x['text']]}
        )

        # Load and preprocess Flickr Order dataset
        flickr_order_dataset = load_dataset(
            "yjkimstats/ARO_Flickr30K_Order_fmt", trust_remote_code=True, split=self.split
        )
        flickr_order_dataset = flickr_order_dataset.map(
            lambda x: {'text': x['text'], 'hard_texts': x['hard_texts']}
        )

        # Load and preprocess COCO Order dataset
        coco_order_dataset = load_dataset(
            "yjkimstats/ARO_COCO_Order_fmt", trust_remote_code=True, split=self.split
        )
        coco_order_dataset = coco_order_dataset.map(
            lambda x: {'text': x['text'], 'hard_texts': x['hard_texts']}
        )

        # Return the combined DatasetDict
        return DatasetDict(
            {
                'VG_Attribute': vg_attribute_dataset.cast(self.features),
                'VG_Relation': vg_relation_dataset.cast(self.features),
                'Flickr_Order': flickr_order_dataset,
                'COCO_Order': coco_order_dataset,
            }
        )