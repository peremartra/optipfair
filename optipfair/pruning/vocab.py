"""
Shrinks a model's vocabulary to the tokens a target domain actually uses.
The pipeline is:

1. :func:`analyse_vocab_usage` — tokenize a calibration corpus and count how
   often every token id appears.
2. :func:`score_tokens` — rank tokens with a pluggable method from
   :data:`TOKEN_SCORING_METHODS` (``"frequency"``, ``"document_frequency"``)
   or any callable ``(VocabScoringContext) -> Dict[int, float]``.
3. :func:`build_keep_set` — select the original ids to keep: every special /
   added token, the byte-level alphabet (or ``<0xNN>`` byte-fallback tokens),
   plus the ``keep_top_k`` highest-scoring remaining tokens and their merge
   ancestors.
4. :func:`build_pruned_tokenizer` — build a new BPE tokenizer containing only
   the kept tokens (with merges filtered so BPE stays consistent) and new,
   contiguous ids.
5. :func:`reorder_embedding_rows` — shrink the input embeddings (and the
   untied ``lm_head``) to the kept rows, in new-id order.

:func:`prune_model_vocab` runs the whole pipeline in memory and
:func:`save_vocab_pruned` persists the result. The saved directory loads with
plain ``AutoModelForCausalLM.from_pretrained`` / ``AutoTokenizer.from_pretrained``.

Unlike width or depth pruning, vocabulary pruning changes the token-id space,
so the pruned model **must** be used with the returned tokenizer.

Limitation: :func:`optipfair.distill_model` feeds the same ``input_ids`` to
student and teacher and compares full-width logits, so it does not yet support
distilling a vocab-pruned student from a full-vocabulary teacher. The
``new2old`` mapping (also written to ``vocab_mapping.json``) is what such a
setup needs: teacher ids are ``new2old[student_ids]`` and teacher logits are
``teacher_logits[..., new2old]``.

Only fast tokenizers backed by a BPE model are supported (GPT-2 / Llama-3 /
Qwen / Mistral style). WordPiece and Unigram tokenizers raise ``ValueError``.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import torch
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import ByteLevel
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from .utils import count_parameters

logger = logging.getLogger(__name__)

# Names of ``<0xNN>`` byte-fallback tokens (Llama-2 / Mistral style BPE).
_BYTE_FALLBACK_TOKENS = ["<0x{:02X}>".format(b) for b in range(256)]

# Config fields holding token ids that must be remapped to the new id space.
_TOKEN_ID_CONFIG_FIELDS = ("bos_token_id", "eos_token_id", "pad_token_id")

# Text fields tried, in order, when a calibration row is a dict.
_TEXT_FIELDS = ("text", "calibration_text", "content")
_PROMPT_FIELDS = ("problem", "question", "prompt", "query", "instruction")
_RESPONSE_FIELDS = ("reasoning_trace", "solution", "response", "answer", "output")

_TOKENIZE_BATCH_SIZE = 256


@dataclass
class VocabScoringContext:
    """Everything a token scoring method may use.

    Attributes:
        tokenizer: Original tokenizer of the model.
        model: The model being pruned (None when scoring without a model).
        frequency: ``token_id -> total occurrences`` in the calibration corpus.
        document_frequency: ``token_id -> number of documents containing it``.
        num_documents: Number of calibration documents.
    """

    tokenizer: Any
    model: Optional[PreTrainedModel]
    frequency: Dict[int, int]
    document_frequency: Dict[int, int]
    num_documents: int


# A scoring method maps the context to ``token_id -> score``. Higher scores
# are kept first; ids missing from the result score 0.
TokenScorer = Callable[[VocabScoringContext], Dict[int, float]]


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #


def _load_backend_spec(tokenizer: Any) -> Dict[str, Any]:
    """Return the ``tokenizer.json`` spec of a fast BPE tokenizer.

    Raises:
        ValueError: If the tokenizer is not a fast tokenizer or is not BPE.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError(
            "Vocabulary pruning requires a fast tokenizer (PreTrainedTokenizerFast). "
            "Load it with AutoTokenizer.from_pretrained(..., use_fast=True)."
        )
    spec = json.loads(backend.to_str())
    model_type = spec.get("model", {}).get("type")
    if model_type != "BPE":
        raise ValueError(
            f"Vocabulary pruning only supports BPE tokenizers, got model type {model_type!r}."
        )
    return spec


def _post_processor_token_ids(processor: Optional[Dict[str, Any]]) -> List[int]:
    """Collect the token ids hard-coded in a serialized post-processor."""
    if not processor:
        return []
    ptype = processor.get("type")
    ids: List[int] = []
    if ptype == "TemplateProcessing":
        for entry in processor.get("special_tokens", {}).values():
            ids.extend(entry.get("ids", []))
    elif ptype in ("BertProcessing", "RobertaProcessing"):
        for key in ("sep", "cls"):
            if processor.get(key):
                ids.append(processor[key][1])
    elif ptype == "Sequence":
        for sub in processor.get("processors", []):
            ids.extend(_post_processor_token_ids(sub))
    return ids


def _remap_post_processor(processor: Optional[Dict[str, Any]], old2new: Dict[int, int]) -> None:
    """Rewrite, in place, the token ids hard-coded in a serialized post-processor.

    Without this, a ``TemplateProcessing`` that prepends BOS (Llama-3) would
    keep emitting the *original* BOS id after pruning.
    """
    if not processor:
        return
    ptype = processor.get("type")
    if ptype == "TemplateProcessing":
        for entry in processor.get("special_tokens", {}).values():
            entry["ids"] = [old2new[i] for i in entry.get("ids", [])]
    elif ptype in ("BertProcessing", "RobertaProcessing"):
        for key in ("sep", "cls"):
            if processor.get(key):
                token, tid = processor[key]
                processor[key] = [token, old2new[tid]]
    elif ptype == "Sequence":
        for sub in processor.get("processors", []):
            _remap_post_processor(sub, old2new)


def _split_merge(merge: Any) -> Tuple[str, str]:
    """Merges are serialized as [a, b] (tokenizers >= 0.20) or "a b" (older)."""
    if isinstance(merge, str):
        a, b = merge.split(" ", 1)
        return a, b
    return merge[0], merge[1]


def _merge_result(a: str, b: str, prefix: str) -> str:
    """Token produced by merging ``a`` and ``b`` (strips b's subword prefix)."""
    return a + (b[len(prefix) :] if prefix and b.startswith(prefix) else b)


def _merge_parents(model_spec: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    """Map every merge-produced token to the pair it is first built from."""
    prefix = model_spec.get("continuing_subword_prefix") or ""
    parents: Dict[str, Tuple[str, str]] = {}
    for merge in model_spec.get("merges", []):
        a, b = _split_merge(merge)
        parents.setdefault(_merge_result(a, b, prefix), (a, b))
    return parents


def _close_over_merges(keep: set, model_spec: Dict[str, Any], vocab: Dict[str, int]) -> set:
    """Add every merge ancestor of the kept tokens.

    BPE can only produce a token if all the intermediate tokens of its merge
    chain are still in the vocabulary (``hen`` needs ``h`` + ``en``). Those
    intermediates are often rare as *final* tokens, so score ranking
    alone would drop them and silently make frequent tokens unreachable.
    """
    id_to_str = {tid: s for s, tid in model_spec["vocab"].items()}
    parents = _merge_parents(model_spec)
    closed = set(keep)
    stack = [id_to_str[i] for i in keep if i in id_to_str]
    while stack:
        token = stack.pop()
        for part in parents.get(token, ()):
            tid = vocab.get(part)
            if tid is not None and tid not in closed:
                closed.add(tid)
                stack.append(part)
    return closed


def _wrap_fast_tokenizer(tokenizer: Any, new_tok: Tokenizer) -> PreTrainedTokenizerFast:
    """Wrap a backend ``Tokenizer`` in a ``PreTrainedTokenizerFast``.

    Special-token *strings*, chat template and common settings are copied
    from the source tokenizer, so ``save_pretrained`` writes a complete
    ``special_tokens_map.json`` / ``tokenizer_config.json``. The backend
    already holds the remapped ids, so HF resolves the new ids from the strings.
    """
    kwargs: Dict[str, Any] = {}
    for attr in PreTrainedTokenizerFast.SPECIAL_TOKENS_ATTRIBUTES:
        value = getattr(tokenizer, attr, None)
        if value:
            kwargs[attr] = value
    for attr in (
        "chat_template",
        "model_max_length",
        "padding_side",
        "truncation_side",
        "clean_up_tokenization_spaces",
    ):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            kwargs[attr] = value
    return PreTrainedTokenizerFast(tokenizer_object=new_tok, **kwargs)


def _row_to_text(row: Any, text_field: Optional[str]) -> str:
    """Extract the text of one calibration row (see :func:`_iter_texts`)."""
    if isinstance(row, str):
        return row
    if isinstance(row, dict):
        if text_field is not None:
            return str(row.get(text_field) or "")
        for field in _TEXT_FIELDS:
            if row.get(field):
                return str(row[field])
        prompt = next((row[f] for f in _PROMPT_FIELDS if row.get(f)), "")
        response = next((row[f] for f in _RESPONSE_FIELDS if row.get(f)), "")
        return f"{prompt}\n{response}".strip()
    if isinstance(row, (list, tuple)):
        return "\n".join(str(part) for part in row)
    return str(row)


def _iter_texts(dataset: Any, text_field: Optional[str] = None) -> Iterator[str]:
    """Yield the non-empty texts of a calibration dataset.

    Accepts an iterable of ``str``, of ``dict`` rows (HF ``Dataset`` rows are
    dicts), or of ``(prompt, response)`` pairs. A ``DatasetDict`` is accepted
    if it has a single split, or a ``vocabulary_calibration`` split.

    For dict rows, ``text_field`` is used when given. Otherwise the first
    non-empty of ``text`` / ``calibration_text`` / ``content`` is used, and as a
    fallback a prompt field (``problem``, ``question``, ``prompt``, ...) is
    joined with a response field (``reasoning_trace``, ``solution``,
    ``response``, ...). Include the generated side of your data: the model
    must be able to *produce* those tokens, not only read them.
    """
    rows: Any = dataset
    if isinstance(dataset, dict):
        if "vocabulary_calibration" in dataset:
            rows = dataset["vocabulary_calibration"]
        elif len(dataset) == 1:
            rows = next(iter(dataset.values()))
        else:
            raise ValueError(
                f"calibration_dataset has several splits {sorted(dataset)}; pass a single split."
            )

    for row in rows:
        text = _row_to_text(row, text_field)
        if text:
            yield text


def _batched(items: Iterable[str], size: int) -> Iterator[List[str]]:
    batch: List[str] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _always_keep_ids(tokenizer: Any) -> set:
    """Ids that must survive pruning regardless of their score."""
    keep = set(tokenizer.all_special_ids)
    keep.update(tokenizer.get_added_vocab().values())
    return keep


def _config_token_ids(model: PreTrainedModel) -> List[int]:
    """Token ids referenced by ``model.config`` / ``model.generation_config``."""
    ids: List[int] = []
    configs = [model.config, getattr(model, "generation_config", None)]
    for cfg in configs:
        if cfg is None:
            continue
        for field in _TOKEN_ID_CONFIG_FIELDS:
            value = getattr(cfg, field, None)
            if isinstance(value, int):
                ids.append(value)
            elif isinstance(value, (list, tuple)):
                ids.extend(v for v in value if isinstance(v, int))
    return ids


def _embeddings_tied(model: PreTrainedModel) -> bool:
    """True if ``lm_head`` shares its weight tensor with the input embeddings."""
    head = model.get_output_embeddings()
    if head is None:
        return True  # nothing separate to resize
    return head.weight is model.get_input_embeddings().weight


# ------------------------------------------------------------------ #
# Public: Step 1 — Vocabulary usage analysis
# ------------------------------------------------------------------ #


def analyse_vocab_usage(
    tokenizer: Any,
    dataset: Any,
    text_field: Optional[str] = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Count how often every token id appears in a calibration corpus.

    Vocabulary pruning only costs (almost) nothing if the dropped tokens are
    genuinely never needed, so tokenize a large, representative sample of the
    target domain, including the outputs the model is expected to generate.

    Args:
        tokenizer: Hugging Face tokenizer of the model to prune.
        dataset: Calibration data. An iterable of ``str``, of dict rows, or a
            ``datasets.Dataset``. See :func:`_iter_texts` for the accepted row
            formats.
        text_field: Optional dict key holding the text of each row.
        show_progress: Whether to show a progress bar.

    Returns:
        Dict with ``vocab_size`` (``len(tokenizer)``), ``num_documents``,
        ``used_ids`` (sorted ids seen in the corpus plus special/added ids)
        ``frequency`` (``str(id) -> total occurrences``) and
        ``document_frequency`` (``str(id) -> number of documents containing
        it``). Keys are strings so the dict is JSON-serializable.
    """
    counter: Counter = Counter()
    doc_counter: Counter = Counter()
    n_docs = 0
    texts = _iter_texts(dataset, text_field)
    for batch in tqdm(
        _batched(texts, _TOKENIZE_BATCH_SIZE),
        desc="Counting token usage",
        unit="batch",
        disable=not show_progress,
    ):
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            counter.update(ids)
            doc_counter.update(set(ids))
        n_docs += len(batch)

    used_ids = set(counter) | _always_keep_ids(tokenizer)
    vocab_size = len(tokenizer)

    logger.info(f"Documents processed  : {n_docs}")
    logger.info(f"Tokenizer vocab size : {vocab_size}")
    logger.info(
        f"Distinct tokens used : {len(used_ids)} "
        f"({100 * len(used_ids) / max(vocab_size, 1):.1f}% of vocab)"
    )

    return {
        "vocab_size": vocab_size,
        "num_documents": n_docs,
        "used_ids": sorted(used_ids),
        "frequency": {str(k): v for k, v in counter.items()},
        "document_frequency": {str(k): v for k, v in doc_counter.items()},
    }


# ------------------------------------------------------------------ #
# Public: Step 2a — Token scoring methods
# ------------------------------------------------------------------ #


def score_tokens_by_frequency(ctx: VocabScoringContext) -> Dict[int, float]:
    """Score = total number of occurrences in the calibration corpus."""
    return dict(ctx.frequency)


def score_tokens_by_document_frequency(ctx: VocabScoringContext) -> Dict[int, float]:
    """Score = number of calibration documents containing the token.

    Less sensitive than raw frequency to a token repeated many times in a few
    documents (tables, code dumps, boilerplate).
    """
    return dict(ctx.document_frequency)


# Registry of built-in token selection methods, by name (see
# ``IMPORTANCE_FUNCTIONS`` in mlp_glu.py for the same pattern).
TOKEN_SCORING_METHODS: Dict[str, TokenScorer] = {
    "frequency": score_tokens_by_frequency,
    "document_frequency": score_tokens_by_document_frequency,
}


def _resolve_token_scorer(method: Union[str, TokenScorer]) -> TokenScorer:
    if callable(method):
        return method
    if method in TOKEN_SCORING_METHODS:
        return TOKEN_SCORING_METHODS[method]
    raise ValueError(
        f"Invalid token selection method: {method!r}. "
        f"Choose from {list(TOKEN_SCORING_METHODS)} or pass a callable."
    )


def score_tokens(
    method: Union[str, TokenScorer],
    ctx: VocabScoringContext,
) -> Dict[int, float]:
    """Run a token scoring method and validate its output.

    Args:
        method: Name in :data:`TOKEN_SCORING_METHODS` or a callable
            ``(VocabScoringContext) -> Dict[int, float]``.
        ctx: Scoring inputs.

    Returns:
        ``token_id -> score`` with int keys and float values.

    Raises:
        ValueError: If the method is unknown or returns something other than
            a mapping of token ids to finite numbers.
    """
    scorer = _resolve_token_scorer(method)
    raw = scorer(ctx)
    if not isinstance(raw, dict):
        raise ValueError(f"Token scoring method must return a dict, got {type(raw).__name__}.")
    scores: Dict[int, float] = {}
    for tid, value in raw.items():
        value = float(value)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"Token scoring method returned a non-finite score for id {tid}.")
        scores[int(tid)] = value
    return scores


# ------------------------------------------------------------------ #
# Public: Step 2b — Keep-set selection (score-driven)
# ------------------------------------------------------------------ #


def build_keep_set(
    tokenizer: Any,
    scores: Dict[int, float],
    keep_top_k: int,
    extra_buffer_ids: Optional[Iterable[int]] = None,
) -> List[int]:
    """Select the sorted list of original token ids to keep.

    Always kept:
        - every special / added token;
        - the base byte tokens: the ``ByteLevel`` alphabet (GPT-2 / Llama-3 /
          Qwen style) and the ``<0xNN>`` byte-fallback tokens (Llama-2 /
          Mistral style), when present in the vocab. They make any input
          losslessly encodable, even text the calibration corpus never had;
        - ids hard-coded in the tokenizer's post-processor (e.g. BOS);
        - ``extra_buffer_ids``, if given.

    On top of that, the ``keep_top_k`` highest-scoring remaining tokens are
    kept (see :func:`score_tokens`). ``keep_top_k`` counts tokens *in
    addition to* the always-kept set. Tokens without a score count as 0, and
    ties are broken by lowest id, so when ``keep_top_k`` exceeds the number of
    scored tokens the remaining slots are filled with unscored ones.

    Finally, the set is closed over BPE merges: every intermediate token
    needed to build a kept token is kept too, otherwise BPE could no longer
    produce it and the pruned tokenizer would split text differently. So
    ``len(result) >= len(always_keep) + keep_top_k`` (capped at the vocab
    size); the extra ``merge_ancestors`` count is logged.

    Args:
        tokenizer: Fast BPE tokenizer of the model.
        scores: Mapping ``token_id -> score`` (int keys, higher is kept first),
            e.g. raw frequency counts or the output of :func:`score_tokens`.
        keep_top_k: Number of score-ranked tokens to keep beyond the
            always-kept set.
        extra_buffer_ids: Optional additional ids to force-keep.

    Returns:
        Sorted list of original token ids to keep.

    Raises:
        ValueError: If ``keep_top_k`` is negative or ``extra_buffer_ids``
            contains ids that do not exist in the tokenizer.
    """
    if keep_top_k < 0:
        raise ValueError(f"keep_top_k must be >= 0, got {keep_top_k}.")

    spec = _load_backend_spec(tokenizer)
    vocab = tokenizer.get_vocab()  # str -> int, includes added tokens
    valid_ids = set(vocab.values())

    always_keep = _always_keep_ids(tokenizer)
    always_keep.update(_post_processor_token_ids(spec.get("post_processor")))

    byte_alphabet = ByteLevel.alphabet()
    byte_level_present = [b for b in byte_alphabet if b in vocab]
    # Only treat the ByteLevel alphabet as byte tokens when the tokenizer is
    # actually byte-level (most of the alphabet present); otherwise these are
    # just ordinary single characters ranked by score.
    if len(byte_level_present) >= len(byte_alphabet) // 2:
        always_keep.update(vocab[b] for b in byte_level_present)
        missing = len(byte_alphabet) - len(byte_level_present)
        if missing:
            logger.info(f"[keep_set] {missing} ByteLevel alphabet tokens not in vocab, skipped")
    always_keep.update(vocab[b] for b in _BYTE_FALLBACK_TOKENS if b in vocab)

    if extra_buffer_ids is not None:
        extra = set(int(i) for i in extra_buffer_ids)
        unknown = extra - valid_ids
        if unknown:
            raise ValueError(
                f"extra_buffer_ids contains ids not in the tokenizer: {sorted(unknown)[:10]}"
            )
        always_keep |= extra

    # Rank the remaining vocabulary by descending score; unscored tokens get
    # 0 and ties are broken by lowest id.
    candidates = valid_ids - always_keep
    ranked = sorted(candidates, key=lambda tid: (-scores.get(tid, 0.0), tid))
    top_k = set(ranked[:keep_top_k])

    selected = always_keep | top_k
    keep = _close_over_merges(selected, spec["model"], vocab)
    scored = {tid for tid, v in scores.items() if v > 0}
    logger.info(
        f"[keep_set] always_keep={len(always_keep)} scored={len(scored)} "
        f"candidates={len(candidates)} kept_scored={len(top_k & scored)} "
        f"kept_fill={len(top_k - scored)} merge_ancestors={len(keep) - len(selected)} "
        f"total_keep={len(keep)}"
    )
    return sorted(keep)


# ------------------------------------------------------------------ #
# Public: Step 2c — Build pruned tokenizer (filtered merges)
# ------------------------------------------------------------------ #


def build_pruned_tokenizer(
    tokenizer: Any,
    keep_ids_asc: List[int],
) -> Tuple[Tokenizer, torch.Tensor]:
    """Build a backend ``Tokenizer`` whose vocabulary is exactly ``keep_ids_asc``.

    New ids follow ascending original-id order, so specials and the byte
    alphabet stay near the bottom of the id space, as in the original layout.
    A merge ``(a, b)`` is kept only if ``a``, ``b`` and their merge result are
    all kept, otherwise BPE could produce a dropped token. Added tokens,
    post-processor ids and the padding id are renumbered into the new space.

    Args:
        tokenizer: Fast BPE tokenizer of the model.
        keep_ids_asc: Sorted original ids to keep (see :func:`build_keep_set`).
            Must include every added / special token.

    Returns:
        ``(new_tokenizer, new2old)``. ``new2old`` is a ``LongTensor`` where
        ``new2old[new_id] == original_id``.

    Raises:
        ValueError: If the tokenizer is unsupported, or ``keep_ids_asc``
            drops a token that the tokenizer's added tokens or post-processor
            need.
    """
    spec = _load_backend_spec(tokenizer)
    model_spec = spec["model"]
    old_vocab: Dict[str, int] = model_spec["vocab"]
    id_to_str = {tid: s for s, tid in old_vocab.items()}
    for added in spec["added_tokens"]:
        id_to_str[added["id"]] = added["content"]

    unknown = [i for i in keep_ids_asc if i not in id_to_str]
    if unknown:
        raise ValueError(f"keep_ids_asc contains ids not in the tokenizer: {unknown[:10]}")

    new2old = torch.tensor(keep_ids_asc, dtype=torch.long)
    old2new = {int(orig): new for new, orig in enumerate(keep_ids_asc)}

    required = [a["id"] for a in spec["added_tokens"]]
    required += _post_processor_token_ids(spec.get("post_processor"))
    missing = sorted(set(i for i in required if i not in old2new))
    if missing:
        raise ValueError(
            f"keep_ids_asc must include all added/special tokens; missing ids {missing[:10]}."
        )

    # Added tokens that live only in `added_tokens` (not in the BPE vocab) stay
    # out of the BPE vocab. Ones also present in `model.vocab` stay in both, so
    # the BPE id space keeps no holes and id -> token decoding stays aligned.
    added_only = {a["content"] for a in spec["added_tokens"]} - set(old_vocab)
    new_vocab = {
        id_to_str[orig]: old2new[orig] for orig in keep_ids_asc if id_to_str[orig] not in added_only
    }

    # Filter merges. They are serialized either as [a, b] pairs (tokenizers
    # >= 0.20) or as "a b" strings (older files); keep the original format.
    prefix = model_spec.get("continuing_subword_prefix") or ""
    kept = set(new_vocab)
    kept_merges = []
    for merge in model_spec.get("merges", []):
        a, b = _split_merge(merge)
        if a in kept and b in kept and _merge_result(a, b, prefix) in kept:
            kept_merges.append(merge)

    model_spec["vocab"] = new_vocab
    model_spec["merges"] = kept_merges
    if model_spec.get("unk_token") and model_spec["unk_token"] not in kept:
        logger.warning(f"unk_token {model_spec['unk_token']!r} was pruned; unsetting it")
        model_spec["unk_token"] = None

    for added in spec["added_tokens"]:
        added["id"] = old2new[added["id"]]
    _remap_post_processor(spec.get("post_processor"), old2new)
    padding = spec.get("padding")
    if padding and padding.get("pad_id") is not None and padding["pad_id"] in old2new:
        padding["pad_id"] = old2new[padding["pad_id"]]

    logger.info(
        f"[tokenizer] vocab {len(id_to_str)} -> {len(keep_ids_asc)}, {len(kept_merges)} merges kept"
    )
    return Tokenizer.from_str(json.dumps(spec)), new2old


# ------------------------------------------------------------------ #
# Public: Step 3 — Shrink embedding rows
# ------------------------------------------------------------------ #


def reorder_embedding_rows(
    model: PreTrainedModel,
    new2old: torch.Tensor,
) -> None:
    """Shrink the embedding table to ``len(new2old)`` rows, in place.

    Row ``i`` of the new table is row ``new2old[i]`` of the old one.

    - Tied embeddings: ``resize_token_embeddings`` re-ties the weights, so a
      single ``[N, hidden]`` tensor is shared by the input embeddings and
      ``lm_head``.
    - Untied embeddings: the same row selection is copied into ``lm_head``
      (and its bias, if present).

    ``bos/eos/pad_token_id`` in ``model.config`` and ``model.generation_config``
    are remapped into the new id space. An id that was not kept is set to
    ``None`` with a warning.

    Raises:
        ValueError: If ``new2old`` references rows beyond the embedding table.
    """
    n = int(new2old.shape[0])
    embed = model.get_input_embeddings()
    num_rows = embed.weight.shape[0]
    if n == 0 or int(new2old.max()) >= num_rows:
        raise ValueError(f"new2old must be non-empty and index rows < {num_rows} (embedding size).")

    tied = _embeddings_tied(model)
    old2new = {int(orig): new for new, orig in enumerate(new2old.tolist())}

    old_embed = embed.weight.data.clone()
    old_head = old_bias = None
    if not tied:
        head = model.get_output_embeddings()
        old_head = head.weight.data.clone()
        old_bias = head.bias.data.clone() if getattr(head, "bias", None) is not None else None

    model.resize_token_embeddings(n)  # also updates config.vocab_size

    with torch.no_grad():
        new_embed = model.get_input_embeddings().weight
        new_embed.data.copy_(old_embed.index_select(0, new2old.to(old_embed.device)))
        if not tied:
            head = model.get_output_embeddings()
            head.weight.data.copy_(old_head.index_select(0, new2old.to(old_head.device)))
            if old_bias is not None:
                head.bias.data.copy_(old_bias.index_select(0, new2old.to(old_bias.device)))
    del old_embed, old_head, old_bias

    def remap(field: str, old: Any) -> Any:
        if isinstance(old, (list, tuple)):
            new = [old2new[o] for o in old if o in old2new]
            if len(new) != len(old):
                logger.warning(f"Some {field} values {list(old)} were pruned; keeping {new}")
            return new or None
        if old not in old2new:
            logger.warning(f"{field}={old} was pruned; setting it to None")
            return None
        return old2new[old]

    configs = [model.config, getattr(model, "generation_config", None)]
    for cfg in configs:
        if cfg is None:
            continue
        for field in _TOKEN_ID_CONFIG_FIELDS:
            old = getattr(cfg, field, None)
            if old is not None:
                setattr(cfg, field, remap(field, old))


# ------------------------------------------------------------------ #
# Public: end-to-end API
# ------------------------------------------------------------------ #


def prune_model_vocab(
    model: PreTrainedModel,
    tokenizer: Any,
    calibration_dataset: Any,
    keep_top_k: int,
    token_selection_method: Union[str, TokenScorer] = "frequency",
    extra_keep_ids: Optional[Iterable[int]] = None,
    text_field: Optional[str] = None,
    show_progress: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerFast, torch.Tensor]:
    """Prune a model's vocabulary to the tokens used by a calibration corpus.

    The model is modified in place: its embedding table (and untied
    ``lm_head``) shrink to the kept rows, and token-id config fields are
    remapped. The pruned model only works with the returned tokenizer.

    Args:
        model: Causal LM to prune.
        tokenizer: Its fast BPE tokenizer.
        calibration_dataset: Representative domain texts. An iterable of
            ``str``, of dict rows, or a ``datasets.Dataset`` (see
            :func:`analyse_vocab_usage`).
        keep_top_k: Number of highest-scoring tokens to keep *in addition to*
            the always-kept set (specials, byte tokens, config token ids).
        token_selection_method: How candidate tokens are ranked. A name from
            :data:`TOKEN_SCORING_METHODS` (``"frequency"``,
            ``"document_frequency"``) or a callable
            ``(VocabScoringContext) -> Dict[int, float]``; higher is kept first.
            Specials, byte tokens and merge ancestors are kept regardless.
        extra_keep_ids: Optional original ids to force-keep.
        text_field: Optional dict key holding the text of each row.
        show_progress: Whether to show a progress bar while counting tokens.

    Returns:
        ``(model, new_tokenizer, new2old)``, where ``new2old[new_id]`` is the
        original id of each kept token. If pruning would keep the whole
        vocabulary, the model and tokenizer are returned unchanged with
        ``new2old = arange(len(tokenizer))``.

    Raises:
        ValueError: On a negative ``keep_top_k``, an unknown
            ``token_selection_method``, an unsupported tokenizer, an empty
            calibration dataset, or a tokenizer larger than the model's
            embedding table.
    """
    if keep_top_k is None or keep_top_k < 0:
        raise ValueError(f"keep_top_k must be a non-negative integer, got {keep_top_k}.")
    _resolve_token_scorer(token_selection_method)  # fail fast on unknown methods
    _load_backend_spec(tokenizer)  # fail fast on unsupported tokenizers

    num_rows = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > num_rows:
        raise ValueError(
            f"Tokenizer has {len(tokenizer)} tokens but the model only has {num_rows} "
            "embedding rows; are they from the same checkpoint?"
        )

    usage = analyse_vocab_usage(
        tokenizer, calibration_dataset, text_field=text_field, show_progress=show_progress
    )
    if usage["num_documents"] == 0:
        raise ValueError("calibration_dataset yielded no text; cannot measure vocabulary usage.")
    ctx = VocabScoringContext(
        tokenizer=tokenizer,
        model=model,
        frequency={int(k): v for k, v in usage["frequency"].items()},
        document_frequency={int(k): v for k, v in usage["document_frequency"].items()},
        num_documents=usage["num_documents"],
    )
    scores = score_tokens(token_selection_method, ctx)

    valid_ids = set(tokenizer.get_vocab().values())
    extra = set(int(i) for i in (extra_keep_ids or []))
    extra.update(i for i in _config_token_ids(model) if i in valid_ids)

    keep_ids = build_keep_set(tokenizer, scores, keep_top_k, extra_buffer_ids=extra)
    if len(keep_ids) >= num_rows:
        logger.warning(
            f"[vocab_prune] keep set ({len(keep_ids)}) covers the whole vocabulary "
            f"({num_rows} rows); model left unchanged."
        )
        return model, tokenizer, torch.arange(len(tokenizer), dtype=torch.long)

    original_vocab = num_rows
    new_backend, new2old = build_pruned_tokenizer(tokenizer, keep_ids)
    new_tokenizer = _wrap_fast_tokenizer(tokenizer, new_backend)

    before = count_parameters(model)
    reorder_embedding_rows(model, new2old)
    after = count_parameters(model)

    method_name = (
        token_selection_method
        if isinstance(token_selection_method, str)
        else getattr(token_selection_method, "__name__", "custom")
    )
    model.config.optipfair_vocab_pruning = {
        "keep_top_k": keep_top_k,
        "token_selection_method": method_name,
        "original_vocab_size": original_vocab,
        "pruned_vocab_size": len(keep_ids),
    }

    logger.info(
        f"Vocabulary {original_vocab} -> {len(keep_ids)} "
        f"({100 * len(keep_ids) / original_vocab:.1f}% kept)"
    )
    logger.info(f"Parameters {before:,} -> {after:,} (-{100 * (1 - after / before):.1f}%)")
    return model, new_tokenizer, new2old


def save_vocab_pruned(
    model: PreTrainedModel,
    tokenizer: Any,
    new2old: torch.Tensor,
    output_path: Union[str, Path],
    keep_top_k: Optional[int] = None,
) -> Path:
    """Save a vocab-pruned model, its tokenizer and ``vocab_mapping.json``.

    ``vocab_mapping.json`` holds ``{"keep_top_k": ..., "new2old": [...]}``,
    the original id of every new id, for downstream use (e.g. distillation
    from a full-vocabulary teacher).

    Returns:
        Resolved path of the output directory.
    """
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    with open(out / "vocab_mapping.json", "w") as f:
        json.dump({"keep_top_k": keep_top_k, "new2old": [int(i) for i in new2old.tolist()]}, f)
    return out.resolve()
