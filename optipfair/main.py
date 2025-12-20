from core.pipeline.compression_pipeline import CompressionPipeline
from core.pipeline.types.pipeline_config import (
    PipelineConfig,
    BenchmarkInferencePerformance,
    BenchmarkModelPerformance,
    ProfileLLM,
)
from core.pipeline.types.prune_config import PruneConfig
from core.compression.pruning.types.attention.gqa_kwargs import (
    GroupedQueryAttentionPrunerKwargs,
)
from core.compression.pruning.types.mlp_w_aligment.kwargs import (
    MLPAlignmentPrunerKwargs,
)
from core.profiling.types.llm.analyze_connections import AnalyzeConnections
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "Explain the theory of relativity in simple terms."

    config = PipelineConfig(
        benchmark_inference_performance=BenchmarkInferencePerformance(
            prompt="explain the theory of relativity in simple terms",
            max_new_tokens=100,
            num_runs=20,
            warmup_runs=2,
            benchmark=True,
        ),
        benchmark_model_performance=BenchmarkModelPerformance(
            benchmark=True, batch_size=2, tests={"lambada"}
        ),
        profile_hardware=True,
        profile_llm=ProfileLLM(
            analyze_connections=AnalyzeConnections(
                input_shape=(1, 512),
                sample_input=lambda: "explain the theory of relativity in simple terms",
            )
        ),
        prune_configs=[
            PruneConfig(
                prune_technique="gqa_attention",
                prune_technique_kwargs=GroupedQueryAttentionPrunerKwargs(
                    num_kv_heads_to_keep=7,
                ),
            ),
            PruneConfig(
                prune_technique="mlp_alignment",
                prune_technique_kwargs=MLPAlignmentPrunerKwargs(
                    alignment=128,
                    prune_percent=0.2,
                ),
            ),
        ],
    )

    pipe = CompressionPipeline(model=model, tokenizer=tokenizer)
    compression_pipe = pipe.run(config)

    with open("pipeline_return_combine.json", "w") as f:
        f.write(compression_pipe.model_dump_json(indent=4))
