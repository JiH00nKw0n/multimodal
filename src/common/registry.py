from typing import TypeVar

Processors = TypeVar("Processors", bound="ProcessorMixin")
Tasks = TypeVar("Tasks", bound="BaseTask")
Models = TypeVar("Models", bound="PreTrainedModel")
ModelConfigs = TypeVar("ModelConfigs", bound="PretrainedConfig")
Trainers = TypeVar("Trainers", bound="Trainer")
Builders = TypeVar("Builders", bound="BaseDatasetBuilder")


class Registry:
    mapping = {
        "builder_name_mapping": {},
        "task_name_mapping": {},
        "processor_name_mapping": {},
        "model_name_mapping": {},
        "model_config_name_mapping": {},
        "trainer_name_mapping": {},
    }

    @classmethod
    def register_processor(cls, name):
        def wrap(processor_cls) -> Processors:
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
    def register_task(cls, name):
        def wrap(task_cls) -> Tasks:
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
        def wrap(trainer_cls) -> Trainers:
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
    def register_model(cls, name):
        def wrap(model_cls) -> Models:
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
        def wrap(config_cls) -> Tasks:
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
        def wrap(builder_cls) -> Builders:
            from src.datasets import BaseDatasetBuilder

            assert issubclass(
                builder_cls, BaseDatasetBuilder
            ), "All tasks must inherit BaseTasks"

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
    def get_trainer_class(cls, name):
        return cls.mapping["trainer_name_mapping"].get(name, None)

    @classmethod
    def list_trainers(cls):
        return sorted(cls.mapping["trainer_name_mapping"].keys())

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
    def list_datasets(cls):
        return sorted(cls.mapping["builder_name_mapping"].keys())


registry = Registry()
