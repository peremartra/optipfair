import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from typing import List
from core.evaluation.model_performance.types.compute_perplexity_for_batch_return import (
    ComputePerplexityForBatchReturn,
)
from core.evaluation.model_performance.types.compute_perplexity_for_dataset_return import (
    ComputePerplexityForDatasetReturn,
)


class ModelPerformanceBenchmarker:
    def _compute_perplexity_for_batch(
        self,
        input_texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
    ) -> ComputePerplexityForBatchReturn:
        """
        Computes perplexity for a single batch of text.

        Args:
            input_texts (List[str]): A list of strings for the batch.
            tokenizer (PreTrainedTokenizerBase): The tokenizer for encoding the text.
            model (PreTrainedModel): The language model to evaluate.

        Returns:
            ComputePerplexityForBatchReturn: A dictionary containing the perplexity
            score for each text in the batch and the mean perplexity for the batch.
        """

        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        input_ids: torch.Tensor = inputs["input_ids"]
        attention_mask: torch.Tensor = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits: torch.Tensor = outputs.logits

        shift_logits: torch.Tensor = logits[:, :-1, :]
        shift_labels: torch.Tensor = input_ids[:, 1:]

        log_probs: torch.Tensor = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        target_log_probs = target_log_probs * attention_mask[:, 1:].to(log_probs.dtype)

        negative_log_likelihood: torch.Tensor = -target_log_probs.sum(
            dim=-1
        ) / attention_mask[:, 1:].sum(dim=-1)
        perplexities: torch.Tensor = torch.exp(negative_log_likelihood)
        mean_perplexity_score: torch.Tensor = torch.mean(perplexities)

        return ComputePerplexityForBatchReturn(
            perplexities=perplexities.tolist(),
            mean_perplexity=mean_perplexity_score.item(),
        )

    def evaluate_perplexity(
        self,
        dataset: Dataset,
        text_column: str,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
    ) -> ComputePerplexityForDatasetReturn:
        """
        Computes perplexity for an entire Hugging Face Dataset by processing it in batches.

        Args:
            dataset (Dataset): The dataset to process.
            text_column (str): The name of the column in the dataset that contains the text.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use for encoding the text.
            model (PreTrainedModel): The language model to evaluate.
            batch_size (int): The number of examples to process in each batch.

        Returns:
            ComputePerplexityForDatasetReturn: A dictionary containing a list of perplexity
            scores for each example and the overall mean perplexity across the entire dataset.
        """
        all_perplexities: List[float] = []
        total_examples: int = len(dataset)

        for i in range(0, total_examples, batch_size):
            batch_end: int = min(i + batch_size, total_examples)

            batch_texts: List[str] = dataset[i:batch_end][text_column]

            results = self._compute_perplexity_for_batch(batch_texts, tokenizer, model)
            all_perplexities.extend(results.perplexities)

        mean_perplexity: float = (
            sum(all_perplexities) / len(all_perplexities) if all_perplexities else 0.0
        )

        return ComputePerplexityForDatasetReturn(
            all_perplexities=all_perplexities, mean_perplexity=mean_perplexity
        )
