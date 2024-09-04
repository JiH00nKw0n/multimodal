from typing import TypeVar

ProcessorType = TypeVar("ProcessorType", bound="ProcessorMixin")
CollatorType = TypeVar("CollatorType", bound="BaseCollator")
TaskType = TypeVar("TaskType", bound="BaseTask")
ModelType = TypeVar("ModelType", bound="PreTrainedModel")
ModelConfigType = TypeVar("ModelConfigType", bound="PretrainedConfig")
TrainerType = TypeVar("TrainerType", bound="Trainer")
BuilderType = TypeVar("BuilderType", bound="BaseBuilder")
EvaluatorType = TypeVar("EvaluatorType", bound="BaseEvaluator")


class Registry:
    mapping = {
        "builder_name_mapping": {},
        "task_name_mapping": {},
        "processor_name_mapping": {},
        "collator_name_mapping": {},
        "model_name_mapping": {},
        "model_config_name_mapping": {},
        "trainer_name_mapping": {},
        "evaluator_name_mapping": {},
    }

    @classmethod
    def register_processor(cls, name):
        def wrap(processor_cls) -> ProcessorType:
            from transformers import ProcessorMixin

            assert issubclass(
                processor_cls, ProcessorMixin
            ), "All processors must inherit ProcessorMixin"

            if name in cls.mapping["processor_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["processor_name_mapping"][name]
                    )
                )
            cls.mapping["processor_name_mapping"][name] = processor_cls

            return processor_cls

        return wrap

    @classmethod
    def register_collator(cls, name):
        def wrap(collator_cls) -> CollatorType:
            from src.common.collator import BaseCollator

            assert issubclass(
                collator_cls, BaseCollator
            ), "All processors must inherit ProcessorMixin"

            if name in cls.mapping["collator_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["collator_name_mapping"][name]
                    )
                )
            cls.mapping["collator_name_mapping"][name] = collator_cls

            return collator_cls

        return wrap

    @classmethod
    def register_task(cls, name):
        def wrap(task_cls) -> TaskType:
            from src.tasks import BaseTask

            assert issubclass(
                task_cls, BaseTask
            ), "All tasks must inherit BaseTasks"

            if name in cls.mapping["task_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["task_name_mapping"][name]
                    )
                )
            cls.mapping["task_name_mapping"][name] = task_cls

            return task_cls

        return wrap

    @classmethod
    def register_trainer(cls, name):
        def wrap(trainer_cls) -> TrainerType:
            from transformers import Trainer

            assert issubclass(
                trainer_cls, Trainer
            ), "All trainer must inherit Trainer"

            if name in cls.mapping["trainer_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["trainer_name_mapping"][name]
                    )
                )
            cls.mapping["trainer_name_mapping"][name] = trainer_cls

            return trainer_cls

        return wrap

    @classmethod
    def register_evaluator(cls, name):
        def wrap(evaluator_cls) -> EvaluatorType:
            from src.evaluators.evaluator import BaseEvaluator

            assert issubclass(
                evaluator_cls, BaseEvaluator
            ), "All evaluator must inherit BaseEvaluator"

            if name in cls.mapping["evaluator_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["evaluator_name_mapping"][name]
                    )
                )
            cls.mapping["evaluator_name_mapping"][name] = evaluator_cls

            return evaluator_cls

        return wrap

    @classmethod
    def register_model(cls, name):
        def wrap(model_cls) -> ModelType:
            from transformers import PreTrainedModel

            assert issubclass(
                model_cls, PreTrainedModel
            ), "All tasks must inherit PreTrainedModel"

            if name in cls.mapping["model_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["model_name_mapping"][name]
                    )
                )
            cls.mapping["model_name_mapping"][name] = model_cls

            return model_cls

        return wrap

    @classmethod
    def register_model_config(cls, name):
        def wrap(config_cls) -> ModelConfigType:
            from transformers import PretrainedConfig

            assert issubclass(
                config_cls, PretrainedConfig
            ), "All tasks must inherit BaseTasks"

            if name in cls.mapping["model_config_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["model_config_name_mapping"][name]
                    )
                )
            cls.mapping["model_config_name_mapping"][name] = config_cls

            return config_cls

        return wrap

    @classmethod
    def register_builder(cls, name):
        def wrap(builder_cls) -> BuilderType:
            from src.datasets import BaseBuilder

            assert issubclass(
                builder_cls, BaseBuilder
            ), "All tasks must inherit BaseDatasetBuilder or SequenceTextDatasetBuilder"

            if name in cls.mapping["builder_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["builder_name_mapping"][name]
                    )
                )
            cls.mapping["builder_name_mapping"][name] = builder_cls

            return builder_cls

        return wrap

    @classmethod
    def get_builder_class(cls, name):
        return cls.mapping["builder_name_mapping"].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_model_config_class(cls, name):
        return cls.mapping["model_config_name_mapping"].get(name, None)

    @classmethod
    def get_task_class(cls, name):
        return cls.mapping["task_name_mapping"].get(name, None)

    @classmethod
    def get_processor_class(cls, name):
        return cls.mapping["processor_name_mapping"].get(name, None)

    @classmethod
    def get_collator_class(cls, name):
        return cls.mapping["collator_name_mapping"].get(name, None)

    @classmethod
    def get_trainer_class(cls, name):
        return cls.mapping["trainer_name_mapping"].get(name, None)

    @classmethod
    def get_evaluator_class(cls, name):
        return cls.mapping["evaluator_name_mapping"].get(name, None)

    @classmethod
    def list_trainers(cls):
        return sorted(cls.mapping["trainer_name_mapping"].keys())

    @classmethod
    def list_evaluators(cls):
        return sorted(cls.mapping["evaluator_name_mapping"].keys())

    @classmethod
    def list_models(cls):
        return sorted(cls.mapping["model_name_mapping"].keys())

    @classmethod
    def list_model_configs(cls):
        return sorted(cls.mapping["model_config_name_mapping"].keys())

    @classmethod
    def list_tasks(cls):
        return sorted(cls.mapping["task_name_mapping"].keys())

    @classmethod
    def list_processors(cls):
        return sorted(cls.mapping["processor_name_mapping"].keys())

    @classmethod
    def list_collators(cls):
        return sorted(cls.mapping["collator_name_mapping"].keys())

    @classmethod
    def list_datasets(cls):
        return sorted(cls.mapping["builder_name_mapping"].keys())


registry = Registry()
