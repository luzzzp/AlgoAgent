from __future__ import annotations

import unittest

from training.sft_train import _build_sft_config, _build_sft_trainer_kwargs
from training.dpo_train import _build_dpo_config, _build_dpo_trainer_kwargs


class TrainingCompatTest(unittest.TestCase):
    def test_sft_config_uses_max_length_when_max_seq_length_is_unavailable(self) -> None:
        class NewSFTConfig:
            def __init__(self, output_dir: str, max_length: int, dataset_text_field: str, **kwargs):
                self.output_dir = output_dir
                self.max_length = max_length
                self.dataset_text_field = dataset_text_field
                self.kwargs = kwargs

        config = _build_sft_config(NewSFTConfig, "outputs/test", 2048)

        self.assertEqual(config.output_dir, "outputs/test")
        self.assertEqual(config.max_length, 2048)
        self.assertEqual(config.dataset_text_field, "text")

    def test_sft_trainer_uses_processing_class_for_newer_trl(self) -> None:
        class NewSFTTrainer:
            def __init__(self, model, args, train_dataset, processing_class, peft_config):
                pass

        kwargs = _build_sft_trainer_kwargs(
            NewSFTTrainer,
            model="model",
            training_args="args",
            dataset="dataset",
            tokenizer="tokenizer",
            peft_config="peft",
        )

        self.assertEqual(kwargs["processing_class"], "tokenizer")
        self.assertNotIn("tokenizer", kwargs)

    def test_dpo_config_adds_beta_when_supported(self) -> None:
        class DPOConfigWithBeta:
            def __init__(self, output_dir: str, beta: float, **kwargs):
                self.output_dir = output_dir
                self.beta = beta
                self.kwargs = kwargs

        config = _build_dpo_config(DPOConfigWithBeta, "outputs/dpo")

        self.assertEqual(config.output_dir, "outputs/dpo")
        self.assertEqual(config.beta, 0.1)

    def test_dpo_trainer_omits_peft_config_for_existing_adapter(self) -> None:
        class NewDPOTrainer:
            def __init__(self, model, args, train_dataset, processing_class):
                pass

        kwargs = _build_dpo_trainer_kwargs(
            NewDPOTrainer,
            model="model",
            training_args="args",
            dataset="dataset",
            tokenizer="tokenizer",
            peft_config=None,
        )

        self.assertEqual(kwargs["processing_class"], "tokenizer")
        self.assertNotIn("peft_config", kwargs)


if __name__ == "__main__":
    unittest.main()
