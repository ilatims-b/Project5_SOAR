from typing import List, Dict, Tuple, Any
from abc import abstractmethod
import evaluate
from transformers import PreTrainedModel
import numpy as np
import torch


class BaseMetric:
    @abstractmethod
    def compute(
        self,
        original_generated_texts: List[str],
        compressed_generated_texts: List[str],
        truncated_compressed_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        """
        Returns a dict of metric_name -> score (float) or a dict whose values are floats.
        Example:
            {
               "metric_1": 0.42,
               "rouge_c_rouge1": 0.51,
               ...
            }
        """
        raise NotImplementedError


def _extract_rouge_score(value: Any) -> float:
    """
    Handle multiple possible return types from evaluate('rouge'):
      - float or np.float64
      - dict-like with 'mid' -> {'fmeasure': ...}
      - object with .mid.fmeasure
    """
    # Direct float
    if isinstance(value, (float, np.floating)):
        return float(value)

    # dict-like
    if isinstance(value, dict):
        # sometimes: {'mid': {'precision': x, 'recall': y, 'fmeasure': z}, ...}
        mid = value.get("mid")
        if isinstance(mid, dict) and "fmeasure" in mid:
            return float(mid["fmeasure"])
        # fallback: try common keys directly
        for k in ("fmeasure", "f", "score"):
            if k in value:
                return float(value[k])

    # object with attributes
    mid = getattr(value, "mid", None)
    if mid is not None:
        fm = getattr(mid, "fmeasure", None)
        if fm is not None:
            return float(fm)

    # Last resort: try to cast
    try:
        return float(value)
    except Exception:
        raise TypeError(f"Unrecognized ROUGE result type: {type(value)} -> {value!r}")


class MeteorMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = evaluate.load("meteor")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        score = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )["meteor"]
        return {"meteor": float(score)}


class Rouge_C(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = evaluate.load("rouge")

    def compute(
        self,
        original_generated_texts: List[str],
        compressed_generated_texts: List[str],
        truncated_compressed_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        # Compare compressed outputs to original generated (compression fidelity)
        metric_results = self._metric.compute(
            predictions=compressed_generated_texts,
            references=original_generated_texts,
            use_stemmer=False,
            # use_aggregator left default; handle both float/object returns below
        )
        score_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        metric_dict: Dict[str, float] = {}
        for rouge_type in score_keys:
            metric_dict[f"rouge_c_{rouge_type}"] = _extract_rouge_score(metric_results[rouge_type])
        return metric_dict


class Rouge_R(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = evaluate.load("rouge")

    def compute(
        self,
        original_generated_texts: List[str],
        compressed_generated_texts: List[str],
        truncated_compressed_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        # Compare truncated compressed outputs to references (task relevance)
        metric_results = self._metric.compute(
            predictions=truncated_compressed_texts,
            references=reference_texts,
            use_stemmer=False,
        )
        score_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        metric_dict: Dict[str, float] = {}
        for rouge_type in score_keys:
            metric_dict[f"rouge_r_{rouge_type}"] = _extract_rouge_score(metric_results[rouge_type])
        return metric_dict
