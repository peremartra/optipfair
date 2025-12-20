import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from typing import Set, Literal, List, Dict, Callable, get_args
from core.evaluation.model_performance.types.compute_perplexity_for_batch_return import (
    ComputePerplexityForBatchReturn,
)
from core.evaluation.model_performance.types.compute_perplexity_for_dataset_return import (
    ComputePerplexityForDatasetReturn,
)
from datasets import load_dataset
from core.evaluation.model_performance.types.perplexity_test_result import (
    PerplexityTestResult,
)
from loguru import logger


# if the test have two words, it need to be separated by underscore. Because it will be mapped to methods inside the ModelPerformanceBenchmarker class
PERPLEXITY_TESTS = Literal["lambada"]
ACCURACY_TESTS = Literal["arc_c"]

AVAILABLE_TESTS: List[str] = list(get_args(PERPLEXITY_TESTS) + get_args(ACCURACY_TESTS))


class ModelPerformanceBenchmarker:
    def __init__(self):
        self._test_methods: Dict[
            str,
            Callable[
                [PreTrainedTokenizerBase, PreTrainedModel, int],
                ComputePerplexityForDatasetReturn,
            ],
        ] = {}
        for test in AVAILABLE_TESTS:
            if hasattr(self, test):
                self._test_methods[test] = getattr(self, test)

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
        num_examples: int = 2,
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
        effective_size: int = (
            min(len(dataset), num_examples)
            if num_examples is not None
            else len(dataset)
        )

        for i in range(0, effective_size, batch_size):
            batch_end: int = min(i + batch_size, effective_size)

            batch_texts: List[str] = dataset[i:batch_end][text_column]

            results = self._compute_perplexity_for_batch(batch_texts, tokenizer, model)
            all_perplexities.extend(results.perplexities)

        mean_perplexity: float = (
            sum(all_perplexities) / len(all_perplexities) if all_perplexities else 0.0
        )

        return ComputePerplexityForDatasetReturn(
            all_perplexities=all_perplexities, mean_perplexity=mean_perplexity
        )

    def lambada(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
    ) -> PerplexityTestResult:
        dataset = load_dataset("cimec/lambada")
        result = self.evaluate_perplexity(
            dataset=dataset["test"],
            text_column="text",
            batch_size=batch_size,
            model=model,
            tokenizer=tokenizer,
        )
        return PerplexityTestResult(result=result, test_name="lambada")

    def benchmark(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        tests: Set[Literal[PERPLEXITY_TESTS, ACCURACY_TESTS]],
        batch_size: int = 16,
    ) -> List[PerplexityTestResult]:
        if not tests:
            raise ValueError(
                "Your should inform at least one test to benchmark the model"
            )

        test_results = list()
        for test in tests:
            method = self._test_methods.get(test)
            if method is None:
                logger.warning(f"test with name {test} is not implemented")
                continue

            result = method(tokenizer, model, batch_size)
            test_results.append(result)

        return test_results
