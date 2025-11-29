import time
from typing import Tuple, Any, Optional, Callable, Dict, List
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from core.profiling.types.llm import (
    ArchitectureInfo,
    ModelSummary,
    InferenceTimeInfo,
    LLMInfo,
    ParameterInfo,
    AttentionLayerAnalysisInfo,
    MemoryEstimate,
    MemoryEstimationInfo,
    PrecisionType,
    ConnectionAnalysisInfo,
    LayerConnectionInfo,
    MeasureInferenceTime,
    AnalyzeConnections,
    EstimateMemory,
)


class LLMProfiler:
    """Enhanced LLM profiler with memory estimation capabilities."""

    PRECISION_BYTES = {
        PrecisionType.FP32: 4.0,
        PrecisionType.FP16: 2.0,
        PrecisionType.BFLOAT16: 2.0,
        PrecisionType.INT8: 1.0,
        PrecisionType.INT4: 0.5,
    }

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        model_name: str = "Unknown Model",
        verbose: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.verbose = verbose
        self.device = self._detect_device()
        self.profile_data: Optional[LLMInfo] = None

    def _detect_device(self) -> torch.device:
        """Detect the device where the model is located."""

        first_param = next(iter(self.model.parameters()), None)
        if first_param is not None:
            return first_param.device

        first_buffer = next(iter(self.model.buffers()), None)
        if first_buffer is not None:
            return first_buffer.device

        return torch.device("cpu")

    def count_parameters(self) -> ParameterInfo:
        """Count and categorize model parameters."""
        total = 0
        trainable = 0
        non_trainable = 0
        dtype_counts = defaultdict(int)
        dtype_bytes = defaultdict(int)

        for param in self.model.parameters():
            num_params = param.numel()
            total += num_params

            if param.requires_grad:
                trainable += num_params
            else:
                non_trainable += num_params

            dtype_str = str(param.dtype)
            dtype_counts[dtype_str] += num_params
            dtype_bytes[dtype_str] += num_params * param.element_size()

        return ParameterInfo(
            total=total,
            trainable=trainable,
            non_trainable=non_trainable,
            total_millions=round(total / 1e6, 2),
            total_billions=round(total / 1e9, 3),
            by_dtype_counts=dict(dtype_counts),
            by_dtype_bytes={k: int(v) for k, v in dtype_bytes.items()},
        )

    def estimate_memory_requirements(
        self,
        sequence_length: int = 2048,
        batch_size: int = 1,
        include_kv_cache: bool = True,
        include_activations: bool = True,
        gradient_accumulation_steps: int = 1,
        optimizer_type: str = "adamw",
    ) -> MemoryEstimationInfo:
        if self.verbose:
            logger.info("💾 Estimating memory requirements...")

        param_info = self.count_parameters()
        total_params = param_info.total

        arch_info = self.analyze_architecture()
        attention_info = self.analyze_attention_layers()

        estimates = {}

        for precision_type in PrecisionType:
            bytes_per_param = self.PRECISION_BYTES[precision_type]

            # Base model weights memory
            model_weights_mb = (total_params * bytes_per_param) / (1024**2)

            # KV Cache memory (only for inference)
            kv_cache_mb = None
            if include_kv_cache and attention_info.num_attention_layers > 0:
                kv_cache_mb = self._estimate_kv_cache_memory(
                    attention_info, sequence_length, batch_size, bytes_per_param
                )

            # Activation memory (used in both training and inference)
            activation_memory_mb = None
            if include_activations:
                activation_memory_mb = self._estimate_activation_memory(
                    arch_info, sequence_length, batch_size, bytes_per_param
                )

            # Training-specific memory components
            training_estimates = self._estimate_training_memory(
                total_params,
                bytes_per_param,
                optimizer_type,
                gradient_accumulation_steps,
                precision_type,
            )

            gradient_memory_mb = training_estimates["gradients"]
            optimizer_memory_mb = training_estimates["optimizer"]
            training_overhead_mb = training_estimates["total"]

            # Calculate inference memory
            inference_memory_mb = model_weights_mb
            if kv_cache_mb:
                inference_memory_mb += kv_cache_mb
            if activation_memory_mb:
                inference_memory_mb += activation_memory_mb

            # Apply inference overhead (typically lower than training)
            inference_overhead_multiplier = 1.2
            total_inference_memory_mb = (
                inference_memory_mb * inference_overhead_multiplier
            )

            # Calculate training memory
            training_memory_mb = model_weights_mb
            if activation_memory_mb:
                training_memory_mb += activation_memory_mb
            training_memory_mb += training_overhead_mb

            # Apply training overhead (higher due to additional complexity)
            training_overhead_multiplier = 1.3
            total_training_memory_mb = training_memory_mb * training_overhead_multiplier

            estimates[precision_type.value] = MemoryEstimate(
                precision=precision_type.value,
                bytes_per_parameter=bytes_per_param,
                # Keep original total_memory_mb for backward compatibility (defaults to training)
                total_memory_mb=round(total_training_memory_mb, 2),
                total_memory_gb=round(total_training_memory_mb / 1024, 3),
                model_weights_mb=round(model_weights_mb, 2),
                kv_cache_mb=round(kv_cache_mb, 2) if kv_cache_mb else None,
                activation_memory_mb=(
                    round(activation_memory_mb, 2) if activation_memory_mb else None
                ),
                # Inference estimate
                total_inference_memory_mb=round(total_inference_memory_mb, 2),
                # total_inference_memory_gb=round(total_inference_memory_mb / 1024, 3),
                # Training estimates
                gradient_memory_mb=round(gradient_memory_mb, 2),
                optimizer_memory_mb=round(optimizer_memory_mb, 2),
                training_memory_mb=round(training_overhead_mb, 2),
            )

        return MemoryEstimationInfo(estimates=estimates, base_parameters=total_params)

    def _estimate_training_memory(
        self,
        total_params: int,
        bytes_per_param: int,
        optimizer_type: str,
        gradient_accumulation_steps: int,
        precision_type: PrecisionType,
    ) -> dict:
        """Estimate memory requirements for training."""

        gradient_memory_mb = (total_params * bytes_per_param) / (1024**2)

        optimizer_memory_mb = 0

        if optimizer_type.lower() == "adamw":
            optimizer_memory_mb = (total_params * 8) / (1024**2)
        elif optimizer_type.lower() == "sgd":
            optimizer_memory_mb = (total_params * bytes_per_param) / (1024**2)
        elif optimizer_type.lower() == "adafactor":
            optimizer_memory_mb = (total_params * 4) / (1024**2)
        else:
            optimizer_memory_mb = (total_params * 8) / (1024**2)

        if gradient_accumulation_steps > 1:
            gradient_memory_mb *= gradient_accumulation_steps

        if (
            precision_type == PrecisionType.FP16
            or precision_type == PrecisionType.BFLOAT16
        ):
            master_weights_mb = (total_params * 4) / (1024**2)
            optimizer_memory_mb += master_weights_mb

        total_training_memory_mb = gradient_memory_mb + optimizer_memory_mb

        return {
            "gradients": gradient_memory_mb,
            "optimizer": optimizer_memory_mb,
            "total": total_training_memory_mb,
        }

    def _estimate_kv_cache_memory(
        self,
        attention_info: AttentionLayerAnalysisInfo,
        sequence_length: int,
        batch_size: int,
        bytes_per_param: float,
    ) -> float:
        """Estimate KV cache memory requirements."""
        total_kv_memory = 0

        for layer in attention_info.attention_layers:
            num_heads = layer.get("num_attention_heads") or layer.get("num_heads") or 12

            head_dim = layer.get("head_dim")
            if not head_dim:
                embed_dim = layer.get("embed_dim", 768)
                head_dim = embed_dim // num_heads

            layer_kv_memory = (
                2
                * batch_size
                * num_heads
                * sequence_length
                * head_dim
                * bytes_per_param
            )
            total_kv_memory += layer_kv_memory

        return total_kv_memory / (1024**2)

    def _estimate_activation_memory(
        self,
        arch_info: ArchitectureInfo,
        sequence_length: int,
        batch_size: int,
        bytes_per_param: float,
    ) -> float:
        """Estimate activation memory requirements (rough approximation)."""

        embed_dim = 768
        for layer in arch_info.layer_details:
            if layer.get("embedding_dim"):
                embed_dim = layer["embedding_dim"]
                break
            elif layer.get("output_size"):
                embed_dim = max(embed_dim, layer["output_size"])

        num_layers = len(
            [
                layer
                for layer in arch_info.layer_details
                if "attention" in layer.get("type", "").lower()
            ]
        )
        if num_layers == 0:
            num_layers = 12

        activation_multiplier = 4

        activation_memory_bytes = (
            batch_size
            * sequence_length
            * embed_dim
            * num_layers
            * activation_multiplier
            * bytes_per_param
        )

        return activation_memory_bytes / (1024**2)

    def analyze_architecture(self) -> ArchitectureInfo:
        """Analyze model architecture and extract layer information."""
        layer_types = defaultdict(int)
        layer_details = []
        total_layers = 0

        for name, module in self.model.named_modules():
            if not list(module.children()):
                layer_type = type(module).__name__
                layer_types[layer_type] += 1

                layer_params = sum(p.numel() for p in module.parameters())
                depth = name.count(".") if name else 0

                layer_info = {
                    "name": name if name else layer_type,
                    "type": layer_type,
                    "parameters": layer_params,
                    "depth": depth,
                }

                self._extract_layer_attributes(module, layer_info)

                layer_details.append(layer_info)
                total_layers += 1

        max_depth = max((ld["depth"] for ld in layer_details), default=0)

        return ArchitectureInfo(
            total_layers=total_layers,
            layer_types_count=dict(layer_types),
            layer_details=layer_details,
            max_depth=max_depth,
        )

    def _extract_layer_attributes(self, module: nn.Module, layer_info: dict) -> None:
        """Extract relevant attributes from a layer module."""

        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            layer_info["input_size"] = getattr(module, "in_features", None)
            layer_info["output_size"] = getattr(module, "out_features", None)

        if hasattr(module, "num_embeddings") and hasattr(module, "embedding_dim"):
            layer_info["vocab_size"] = getattr(module, "num_embeddings", None)
            layer_info["embedding_dim"] = getattr(module, "embedding_dim", None)

        for attr in ["num_heads", "num_attention_heads", "head_dim", "embed_dim"]:
            if hasattr(module, attr):
                layer_info[attr] = getattr(module, attr, None)

    def measure_inference_time(
        self,
        input_sample: Optional[Callable[[], Any]] = None,
        num_runs: int = 100,
        warmup_runs: int = 10,
        tokenizer_max_length: int = 512,
    ) -> InferenceTimeInfo:
        """Measure model inference time with proper warmup."""
        self.model.eval()

        for _ in range(warmup_runs):
            input_prompt = input_sample()
            tokenized_input = self.tokenizer(
                input_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer_max_length,
            ).to(self.device)

            with torch.no_grad():
                _ = self.model(**tokenized_input)

        execution_times: List[Dict[str, str | float | int]] = list()

        for i in range(num_runs):
            start_time = time.perf_counter()
            input_prompt = input_sample()
            tokenized_input = self.tokenizer(
                input_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer_max_length,
            ).input_ids.to(self.device)

            with torch.no_grad():
                _ = self.model(tokenized_input)

            end_time = time.perf_counter()
            execution_times.append(
                {
                    "run_number": i,
                    "prompt_used": input_prompt,
                    "execution_time_s": (end_time - start_time) * 1000,
                }
            )

        execution_times_array = np.array(
            [run["execution_time_s"] for run in execution_times]
        )

        return InferenceTimeInfo(
            max_time_ms=np.max(execution_times_array),
            mean_time_ms=np.mean(execution_times_array),
            median_time_ms=np.median(execution_times_array),
            min_time_ms=np.min(execution_times_array),
            runs=len(execution_times_array),
            std_time_ms=np.std(execution_times_array),
        )

    def _prepare_inference_input(
        self,
        input_shape: Optional[Tuple[int, ...]],
        input_sample: Optional[Callable[[], Any]],
    ) -> Any:
        """Prepare input for inference timing."""
        if input_sample is not None:
            inp = input_sample()
            return self._move_to_device(inp)

        if input_shape is None:
            raise ValueError(
                "Either input_shape or input_sample must be provided for inference timing."
            )

        if len(input_shape) == 2:
            batch, seq = input_shape
            return torch.randint(
                0, 1000, (batch, seq), dtype=torch.long, device=self.device
            )
        else:
            return torch.randn(input_shape, device=self.device)

    def _move_to_device(self, inp: Any) -> Any:
        """Move input to the appropriate device."""
        if isinstance(inp, dict):
            return {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in inp.items()
            }
        elif isinstance(inp, (tuple, list)):
            return tuple((v.to(self.device) if torch.is_tensor(v) else v) for v in inp)
        elif torch.is_tensor(inp):
            return inp.to(self.device)
        return inp

    def _run_warmup(self, sample: Any, warmup_runs: int) -> None:
        """Run warmup iterations."""
        with torch.no_grad():
            for _ in range(warmup_runs):
                self._forward_pass(sample)

    def _measure_timing(self, sample: Any, num_runs: int) -> np.ndarray:
        """Measure inference timing."""
        times = []

        with torch.no_grad():
            if self.device.type == "cuda":
                times = self._measure_cuda_timing(sample, num_runs)
            else:
                times = self._measure_cpu_timing(sample, num_runs)

        return np.array(times)

    def _measure_cuda_timing(self, sample: Any, num_runs: int) -> list:
        """Measure timing using CUDA events."""
        times = []
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            self._forward_pass(sample)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            times.append(elapsed_ms / 1000.0)

        return times

    def _measure_cpu_timing(self, sample: Any, num_runs: int) -> list:
        """Measure timing using time.time()."""
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            self._forward_pass(sample)
            end_time = time.time()
            times.append(end_time - start_time)

        return times

    def _forward_pass(self, sample: Any) -> Any:
        """Execute a forward pass with the given input."""
        if isinstance(sample, dict):
            return self.model(**sample)
        elif isinstance(sample, (tuple, list)):
            return self.model(*sample)
        else:
            return self.model(sample)

    def _compute_timing_statistics(self, times: np.ndarray) -> InferenceTimeInfo:
        """Compute timing statistics from measured times."""
        return InferenceTimeInfo(
            mean_time_ms=round(float(np.mean(times) * 1000), 4),
            std_time_ms=round(float(np.std(times) * 1000), 4),
            min_time_ms=round(float(np.min(times) * 1000), 4),
            max_time_ms=round(float(np.max(times) * 1000), 4),
            median_time_ms=round(float(np.median(times) * 1000), 4),
            runs=int(len(times)),
        )

    def analyze_attention_layers(self) -> AttentionLayerAnalysisInfo:
        """Analyze attention layers in the model."""
        attention_candidates = self._find_attention_candidates()
        leaf_candidates = self._filter_leaf_candidates(attention_candidates)
        attention_layers = self._extract_attention_info(
            leaf_candidates, attention_candidates
        )

        return AttentionLayerAnalysisInfo(
            num_attention_layers=len(attention_layers),
            attention_layers=attention_layers,
        )

    def _find_attention_candidates(self) -> Dict[str, nn.Module]:
        """Find modules that look like attention layers."""
        attention_candidates = {}

        for name, module in self.model.named_modules():
            if self._looks_like_attention(module):
                candidate_name = name if name else type(module).__name__
                attention_candidates[candidate_name] = module

        return attention_candidates

    def _looks_like_attention(self, module: nn.Module) -> bool:
        """Check if a module looks like an attention layer."""

        if isinstance(module, torch.nn.MultiheadAttention):
            return True

        attention_attrs = [
            "num_attention_heads",
            "num_heads",
            "all_head_size",
            "head_dim",
        ]
        if any(hasattr(module, attr) for attr in attention_attrs):
            return True

        class_name = type(module).__name__.lower()
        attention_keywords = [
            "attention",
            "attn",
            "selfattn",
            "self_attention",
            "multihead",
        ]
        return any(keyword in class_name for keyword in attention_keywords)

    def _filter_leaf_candidates(self, candidates: Dict[str, nn.Module]) -> list:
        """Filter to get only leaf attention candidates."""
        candidate_names = list(candidates.keys())
        leaf_candidates = []

        for name in candidate_names:
            is_parent = any(
                (other != name) and other.startswith(name + ".")
                for other in candidate_names
            )
            if not is_parent:
                leaf_candidates.append(name)

        return sorted(leaf_candidates)

    def _extract_attention_info(
        self, leaf_candidates: list, candidates: Dict[str, nn.Module]
    ) -> list:
        """Extract information from attention layers."""
        attention_layers = []

        for name in leaf_candidates:
            module = candidates[name]
            info = {"name": name, "type": type(module).__name__}

            attention_attrs = [
                "num_attention_heads",
                "num_heads",
                "embed_dim",
                "head_dim",
            ]
            for attr in attention_attrs:
                if hasattr(module, attr):
                    info[attr] = getattr(module, attr)

            attention_layers.append(info)

        return attention_layers

    def get_model_summary(self) -> ModelSummary:
        """Get basic model summary information."""
        return ModelSummary(
            model_name=self.model_name,
            device=str(self.device),
            model_class=type(self.model).__name__,
            pytorch_version=torch.__version__,
        )

    def profile_complete(
        self,
        measure_inference: Optional[MeasureInferenceTime] = None,
        analyze_connections: Optional[AnalyzeConnections] = None,
        estimate_memory: Optional[EstimateMemory] = None,
    ) -> LLMInfo:
        """
        Perform complete model profiling."""
        if self.verbose:
            logger.info(f"🔍 Profiling {self.model_name}...")

        model_summary = self.get_model_summary()

        if self.verbose:
            logger.info("📊 Counting parameters...")
        parameters_info = self.count_parameters()

        if self.verbose:
            logger.info("🏗️  Analyzing architecture...")
        architecture_info = self.analyze_architecture()

        if self.verbose:
            logger.info("🎯 Analyzing attention layers...")
        attention_layer_info = self.analyze_attention_layers()

        if analyze_connections:
            connection_info = self.analyze_connections(
                input_shape=analyze_connections.input_shape,
                input_sample=analyze_connections.sample_input,
            )

            architecture_info.connections = connection_info

        inference_time_info = None
        if measure_inference:
            if self.verbose:
                logger.info("⏱️  Measuring inference time...")
            inference_time_info = self.measure_inference_time(
                input_sample=measure_inference.input_sample,
                num_runs=measure_inference.num_runs,
                warmup_runs=measure_inference.warmup_runs,
                tokenizer_max_length=measure_inference.tokenizer_max_length,
            )

        memory_info = None
        if estimate_memory:
            memory_info = self.estimate_memory_requirements(
                sequence_length=estimate_memory.sequence_length,
                batch_size=estimate_memory.batch_size,
            )

        llm_info = LLMInfo(
            architecture=architecture_info,
            attention_layers=attention_layer_info,
            inference_time=inference_time_info,
            parameters=parameters_info,
            summary=model_summary,
            memory_estimation=memory_info,
        )

        self.profile_data = llm_info
        if self.verbose:
            logger.info("✅ Profiling complete!")

        return llm_info

    def print_summary(self):
        print(self.profile_data.model_dump_json(indent=4))

    def analyze_connections(
        self,
        sample_input: Optional[Any] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        input_sample: Optional[Callable[[], Any]] = None,
    ) -> ConnectionAnalysisInfo:
        """Analyze neural connections between layers."""
        if self.verbose:
            logger.info("🔗 Analyzing layer connections...")

        if sample_input is None:
            sample_input = self._prepare_inference_input(input_shape, input_sample)

        connections = self._analyze_connections_fx(sample_input)

        if connections is None:
            if self.verbose:
                logger.warning("torch.fx failed, falling back to hooks method...")
            connections = self._analyze_connections_hooks(sample_input)

        if connections is None:
            if self.verbose:
                logger.warning(
                    "All connection analysis methods failed, using basic analysis..."
                )
            connections = self._analyze_connections_basic()

        return connections

    def _analyze_connections_fx(
        self, sample_input: Any
    ) -> Optional[ConnectionAnalysisInfo]:
        """Analyze connections using torch.fx symbolic tracing."""
        try:
            import torch.fx

            traced = torch.fx.symbolic_trace(self.model)

            connection_graph = {}
            node_outputs = {}

            for node in traced.graph.nodes:
                if node.op == "call_module":
                    module = dict(traced.named_modules())[node.target]

                    layer_info = LayerConnectionInfo(
                        name=node.target,
                        type=type(module).__name__,
                        parameters=sum(p.numel() for p in module.parameters()),
                        depth=node.target.count(".") if node.target else 0,
                        input_layers=[],
                        output_layers=[],
                    )

                    self._extract_layer_attributes(module, layer_info.__dict__)

                    connection_graph[node.target] = layer_info
                    node_outputs[node] = node.target

            for node in traced.graph.nodes:
                if node.op == "call_module" and node.target in connection_graph:
                    current_layer = connection_graph[node.target]

                    for arg in node.args:
                        if hasattr(arg, "target") and arg.target in connection_graph:
                            input_layer_name = arg.target
                            current_layer.input_layers.append(input_layer_name)
                            connection_graph[input_layer_name].output_layers.append(
                                node.target
                            )

                    for user in node.users:
                        if hasattr(user, "target") and user.target in connection_graph:
                            output_layer_name = user.target
                            if output_layer_name not in current_layer.output_layers:
                                current_layer.output_layers.append(output_layer_name)

            self._get_layer_shapes_fx(traced, sample_input, connection_graph)

            analysis_info = self._analyze_connection_patterns(
                connection_graph, "torch_fx"
            )

            return analysis_info

        except Exception as e:
            if self.verbose:
                logger.debug(f"torch.fx analysis failed: {str(e)}")
            return None

    def _analyze_connections_hooks(
        self, sample_input: Any
    ) -> Optional[ConnectionAnalysisInfo]:
        """Analyze connections using forward hooks."""
        try:
            connection_graph = {}
            layer_shapes = {}
            execution_order = []

            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:
                    layer_info = LayerConnectionInfo(
                        name=name if name else type(module).__name__,
                        type=type(module).__name__,
                        parameters=sum(p.numel() for p in module.parameters()),
                        depth=name.count(".") if name else 0,
                        input_layers=[],
                        output_layers=[],
                    )

                    self._extract_layer_attributes(module, layer_info.__dict__)
                    connection_graph[layer_info.name] = layer_info

            hooks = []

            def create_hook(layer_name):
                def hook_fn(module, input, output):
                    execution_order.append(layer_name)

                    if isinstance(input, (tuple, list)) and len(input) > 0:
                        if torch.is_tensor(input[0]):
                            layer_shapes[layer_name] = {
                                "input_shape": list(input[0].shape),
                                "output_shape": (
                                    list(output.shape)
                                    if torch.is_tensor(output)
                                    else None
                                ),
                            }

                return hook_fn

            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:
                    layer_name = name if name else type(module).__name__
                    if layer_name in connection_graph:
                        hook = module.register_forward_hook(create_hook(layer_name))
                        hooks.append(hook)

            with torch.no_grad():
                _ = self._forward_pass(sample_input)

            for hook in hooks:
                hook.remove()

            for i, layer_name in enumerate(execution_order):
                if layer_name in connection_graph:
                    connection_graph[layer_name].input_shape = layer_shapes.get(
                        layer_name, {}
                    ).get("input_shape")
                    connection_graph[layer_name].output_shape = layer_shapes.get(
                        layer_name, {}
                    ).get("output_shape")

                    if i > 0:
                        prev_layer = execution_order[i - 1]
                        if (
                            prev_layer in connection_graph
                            and prev_layer
                            not in connection_graph[layer_name].input_layers
                        ):
                            connection_graph[layer_name].input_layers.append(prev_layer)
                            connection_graph[prev_layer].output_layers.append(
                                layer_name
                            )

            analysis_info = self._analyze_connection_patterns(connection_graph, "hooks")
            return analysis_info

        except Exception as e:
            if self.verbose:
                logger.debug(f"Hooks analysis failed: {str(e)}")
            return None

    def _analyze_connections_basic(self) -> ConnectionAnalysisInfo:
        """Basic connection analysis based on naming patterns."""
        connection_graph = {}

        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                layer_info = LayerConnectionInfo(
                    name=name if name else type(module).__name__,
                    type=type(module).__name__,
                    parameters=sum(p.numel() for p in module.parameters()),
                    depth=name.count(".") if name else 0,
                    input_layers=[],
                    output_layers=[],
                )

                self._extract_layer_attributes(module, layer_info.__dict__)
                connection_graph[layer_info.name] = layer_info

        return self._analyze_connection_patterns(connection_graph, "basic")

    def _get_layer_shapes_fx(
        self,
        traced_model,
        sample_input: Any,
        connection_graph: Dict[str, LayerConnectionInfo],
    ):
        """Get input/output shapes for layers using torch.fx."""
        try:
            shape_info = {}

            def create_shape_hook(layer_name):
                def hook_fn(module, input, output):
                    input_shape = None
                    output_shape = None

                    if isinstance(input, (tuple, list)) and len(input) > 0:
                        if torch.is_tensor(input[0]):
                            input_shape = list(input[0].shape)

                    if torch.is_tensor(output):
                        output_shape = list(output.shape)
                    elif isinstance(output, (tuple, list)) and len(output) > 0:
                        if torch.is_tensor(output[0]):
                            output_shape = list(output[0].shape)

                    shape_info[layer_name] = {
                        "input_shape": input_shape,
                        "output_shape": output_shape,
                    }

                return hook_fn

            hooks = []
            for name, module in self.model.named_modules():
                if name in connection_graph:
                    hook = module.register_forward_hook(create_shape_hook(name))
                    hooks.append(hook)

            with torch.no_grad():
                _ = self._forward_pass(sample_input)

            for layer_name, shapes in shape_info.items():
                if layer_name in connection_graph:
                    connection_graph[layer_name].input_shape = shapes["input_shape"]
                    connection_graph[layer_name].output_shape = shapes["output_shape"]

            for hook in hooks:
                hook.remove()

        except Exception as e:
            if self.verbose:
                logger.debug(f"Shape extraction failed: {str(e)}")

    def _analyze_connection_patterns(
        self, connection_graph: Dict[str, LayerConnectionInfo], method: str
    ) -> ConnectionAnalysisInfo:
        """Analyze patterns in the connection graph."""
        total_connections = sum(
            len(layer.output_layers) for layer in connection_graph.values()
        )

        has_skip_connections = any(
            len(layer.input_layers) > 1 for layer in connection_graph.values()
        )

        max_fan_in = max(
            (len(layer.input_layers) for layer in connection_graph.values()), default=0
        )
        max_fan_out = max(
            (len(layer.output_layers) for layer in connection_graph.values()), default=0
        )

        return ConnectionAnalysisInfo(
            total_connections=total_connections,
            connection_graph=connection_graph,
            analysis_method=method,
            has_skip_connections=has_skip_connections,
            max_fan_in=max_fan_in,
            max_fan_out=max_fan_out,
        )

    def get_connection_summary(self) -> Optional[Dict[str, Any]]:
        """Get a summary of model connections."""
        if not self.profile_data or not self.profile_data.architecture.connections:
            return None

        conn_info = self.profile_data.architecture.connections

        return {
            "total_connections": conn_info.total_connections,
            "analysis_method": conn_info.analysis_method,
            "has_skip_connections": conn_info.has_skip_connections,
            "max_fan_in": conn_info.max_fan_in,
            "max_fan_out": conn_info.max_fan_out,
            "layers_with_multiple_inputs": [
                name
                for name, layer in conn_info.connection_graph.items()
                if len(layer.input_layers) > 1
            ],
            "layers_with_multiple_outputs": [
                name
                for name, layer in conn_info.connection_graph.items()
                if len(layer.output_layers) > 1
            ],
        }
