"""Familia experimental para OpenThinker3-7B como razonador externo.

Esta familia no sustituye el backbone. Solo deja evidencia estructurada para
perfiles explicitos y benchmarks de laboratorio.
"""

from __future__ import annotations

import json
from json import JSONDecodeError
import os
import re
from typing import Any, Dict, Mapping

from runtime.reasoning.contracts import FamilyResult
from runtime.reasoning.external_models import LlamaCppClient


FAMILY_ID = "EXT_OPEN_THINKER"
_RAW_EXCERPT_LIMIT = 700
_LIST_FIELDS = ("candidate_hypotheses", "causal_assumptions", "counterfactual_checks")
_REQUIRED_FIELDS = (
    "claim",
    "reasoning_summary",
    "candidate_hypotheses",
    "causal_assumptions",
    "counterfactual_checks",
    "recommended_intervention",
    "confidence_proxy",
)
_ALLOWED_FIELDS = set(_REQUIRED_FIELDS)
_MAX_TEXT_LENGTHS = {
    "claim": 240,
    "reasoning_summary": 420,
}
_MAX_LIST_ITEMS = 5
_MAX_LIST_ITEM_LENGTH = 180


class ExternalReasonerParseError(ValueError):
    """Error estructurado de contrato para la salida del razonador externo."""

    def __init__(self, code: str, message: str | None = None):
        super().__init__(message or code)
        self.code = code


def _clamp(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(numeric, 0.0), 1.0)


def _safe_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_list_of_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _allowed_interventions(state: Mapping[str, Any]) -> list[str]:
    metadata = _safe_mapping(state.get("scenario_metadata"))
    interventions = metadata.get("interventions")
    if isinstance(interventions, list):
        return [str(item) for item in interventions if str(item)]
    return ["activate_cooling", "deactivate_cooling"]


def _collect_core_hypotheses(state: Mapping[str, Any]) -> Dict[str, Any]:
    keys = [
        "abd_hypothesis",
        "ana_mapping",
        "cau_link",
        "ctf_checked",
        "ded_conclusion",
        "ded_validated",
        "prob_calibrated",
    ]
    return {key: state.get(key) for key in keys if key in state}


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _strip_think_blocks(text: str) -> str:
    without_closed = re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S)
    return re.sub(r"</?think>", "", without_closed, flags=re.I)


def _balanced_json_objects(text: str) -> list[str]:
    payload = _strip_think_blocks(text)
    objects: list[str] = []
    start = payload.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(payload)):
            char = payload[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    objects.append(payload[start : index + 1])
                    start = payload.find("{", index + 1)
                    break
        else:
            start = payload.find("{", start + 1)
            continue
    return objects


def _looks_like_echoed_prompt(obj: Mapping[str, Any]) -> bool:
    prompt_keys = {
        "instructions",
        "question",
        "schema",
        "allowed_interventions",
        "observation",
        "task",
        "required_schema",
    }
    return bool(prompt_keys.intersection(obj)) and not _ALLOWED_FIELDS.issubset(set(obj))


def _json_from_text(text: str) -> Dict[str, Any]:
    candidates = _balanced_json_objects(text)
    if not candidates:
        raise ExternalReasonerParseError("no_json_object_found")
    first_decode_error: JSONDecodeError | None = None
    contract_markers = {
        "claim",
        "reasoning_summary",
        "candidate_hypotheses",
        "recommended_intervention",
        "confidence_proxy",
    }
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except JSONDecodeError as exc:
            if first_decode_error is None:
                first_decode_error = exc
            continue
        if not isinstance(obj, dict):
            raise ExternalReasonerParseError("schema_validation_error", "top_level_not_object")
        if _looks_like_echoed_prompt(obj):
            continue
        if not contract_markers.intersection(obj):
            continue
        return obj
    if first_decode_error is not None:
        raise ExternalReasonerParseError("json_decode_error", str(first_decode_error))
    raise ExternalReasonerParseError("no_json_object_found")


def _validate_payload_shape(payload: Mapping[str, Any], *, allowed_interventions: list[str]) -> Dict[str, Any]:
    unknown = sorted(set(payload) - _ALLOWED_FIELDS)
    if unknown:
        raise ExternalReasonerParseError(
            "schema_validation_error",
            "unknown_fields:" + ",".join(unknown),
        )
    missing = [field for field in _REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ExternalReasonerParseError(
            "schema_validation_error",
            "missing_fields:" + ",".join(missing),
        )

    normalized: Dict[str, Any] = {}
    for field in ("claim", "reasoning_summary", "recommended_intervention"):
        value = payload.get(field)
        if not isinstance(value, str):
            raise ExternalReasonerParseError("schema_validation_error", f"{field}:not_string")
        value = value.strip()
        if not value:
            raise ExternalReasonerParseError("empty_required_field", field)
        max_length = _MAX_TEXT_LENGTHS.get(field, 80)
        if len(value) > max_length:
            raise ExternalReasonerParseError("schema_validation_error", f"{field}:too_long")
        normalized[field] = value

    for field in _LIST_FIELDS:
        value = payload.get(field)
        if not isinstance(value, list):
            raise ExternalReasonerParseError("schema_validation_error", f"{field}:not_array")
        if len(value) > _MAX_LIST_ITEMS:
            raise ExternalReasonerParseError("schema_validation_error", f"{field}:too_many_items")
        items: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise ExternalReasonerParseError("schema_validation_error", f"{field}[{index}]:not_string")
            item = item.strip()
            if not item:
                raise ExternalReasonerParseError("empty_required_field", f"{field}[{index}]")
            if len(item) > _MAX_LIST_ITEM_LENGTH:
                raise ExternalReasonerParseError("schema_validation_error", f"{field}[{index}]:too_long")
            items.append(item)
        normalized[field] = items

    confidence = payload.get("confidence_proxy")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ExternalReasonerParseError("schema_validation_error", "confidence_proxy:not_number")
    confidence = float(confidence)
    if confidence < 0.0 or confidence > 1.0:
        raise ExternalReasonerParseError("confidence_out_of_range")
    normalized["confidence_proxy"] = confidence

    allowed = {item.strip() for item in allowed_interventions}
    if normalized["recommended_intervention"] not in allowed:
        raise ExternalReasonerParseError("invalid_intervention", normalized["recommended_intervention"])
    return normalized


def _build_standard_prompt(state: Mapping[str, Any]) -> str:
    observation = _safe_mapping(state.get("observation"))
    counterfactual = _safe_mapping(state.get("counterfactual"))
    factual = _safe_mapping(state.get("updated_world"))
    allowed = _allowed_interventions(state)
    compact_observation = {
        "global_temp_mean": observation.get("global_temp_mean", observation.get("world_level")),
        "global_temp_max": observation.get("global_temp_max"),
        "alarm": observation.get("alarm"),
        "propositions": list(observation.get("propositions", []))[:12],
    }
    compact_factual = {
        "intervention": state.get("intervention"),
        "global_temp_mean": factual.get("global_temp_mean", factual.get("world_level")),
        "alarm": factual.get("alarm"),
    }
    compact_counterfactual = {
        "global_temp_mean": counterfactual.get("global_temp_mean", counterfactual.get("world_level")),
        "alarm": counterfactual.get("alarm"),
    }
    prompt_payload = {
        "instructions": [
            "Return only one valid JSON object.",
            "Do not include think tags.",
            "Do not include markdown.",
            "Do not write text before or after JSON.",
            "Use short strings, under 12 words each.",
            "If evidence is insufficient, use empty arrays and low confidence.",
        ],
        "question": "Which allowed intervention is better supported by the observation and counterfactual evidence?",
        "schema": {
            "claim": "short claim",
            "reasoning_summary": "brief evidence summary, no hidden reasoning",
            "candidate_hypotheses": ["string, max 5"],
            "causal_assumptions": ["string, max 5"],
            "counterfactual_checks": ["string, max 5"],
            "recommended_intervention": allowed[0] if allowed else "",
            "confidence_proxy": 0.0,
        },
        "allowed_interventions": allowed,
        "regime": state.get("regime_hint") or state.get("regime_label"),
        "formula": state.get("formula"),
        "observation": compact_observation,
        "core_transition": compact_factual,
        "alternative_transition": compact_counterfactual,
    }
    return json.dumps(prompt_payload, ensure_ascii=True, sort_keys=True)


def _build_compact_prompt(state: Mapping[str, Any]) -> str:
    observation = _safe_mapping(state.get("observation"))
    counterfactual = _safe_mapping(state.get("counterfactual"))
    factual = _safe_mapping(state.get("updated_world"))
    allowed = _allowed_interventions(state)
    prompt_payload = {
        "task": "choose_intervention_json",
        "json_only": True,
        "required_keys": list(_REQUIRED_FIELDS),
        "allowed_interventions": allowed,
        "regime": state.get("regime_hint") or state.get("regime_label"),
        "rule": state.get("formula"),
        "obs": {
            "temp": observation.get("global_temp_mean", observation.get("world_level")),
            "max": observation.get("global_temp_max"),
            "alarm": observation.get("alarm"),
            "props": list(observation.get("propositions", []))[:6],
        },
        "core": {
            "intervention": state.get("intervention"),
            "temp": factual.get("global_temp_mean", factual.get("world_level")),
            "alarm": factual.get("alarm"),
        },
        "alt": {
            "temp": counterfactual.get("global_temp_mean", counterfactual.get("world_level")),
            "alarm": counterfactual.get("alarm"),
        },
        "decision": "prefer lower temp, alarm false, valid intervention, confidence 0..1",
    }
    return json.dumps(prompt_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _build_prompt(state: Mapping[str, Any]) -> str:
    style = str(state.get("external_reasoner_prompt_style") or "standard").strip().lower()
    if style == "compact":
        return _build_compact_prompt(state)
    return _build_standard_prompt(state)


def _prompt_style_for_execution(state: Mapping[str, Any], client: Any) -> str:
    configured = None
    config = getattr(client, "config", None)
    if config is not None:
        configured = getattr(config, "prompt_style", None)
    return str(
        state.get("external_reasoner_prompt_style")
        or configured
        or os.environ.get("RNFE_EXTERNAL_REASONER_PROMPT_STYLE")
        or "standard"
    ).strip().lower()


def parse_external_reasoner_payload(raw_text: str, *, allowed_interventions: list[str]) -> Dict[str, Any]:
    payload = _json_from_text(raw_text)
    return _validate_payload_shape(payload, allowed_interventions=allowed_interventions)


def _result(
    *,
    status: str,
    state_delta: Dict[str, Any],
    confidence: float,
    cost: float,
    artifacts: Dict[str, Any] | None = None,
    failure_mode: str | None = None,
    extras: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = FamilyResult(
        family=FAMILY_ID,
        status=status,
        state_delta=state_delta,
        confidence=confidence,
        cost=cost,
        artifacts=artifacts or {},
        failure_mode=failure_mode,
    ).to_dict()
    if extras:
        payload.update(extras)
    return payload


def execute(state: Dict[str, Any]) -> Dict[str, Any]:
    client = state.get("_external_reasoner_client")
    if client is None:
        client = LlamaCppClient()
    prompt_style = _prompt_style_for_execution(state, client)
    prompt_state = dict(state)
    prompt_state["external_reasoner_prompt_style"] = prompt_style
    prompt = _build_prompt(prompt_state)
    prompt_bytes = len(prompt.encode("utf-8"))

    result = client.generate(
        prompt,
        backend=state.get("external_reasoner_backend"),
        max_tokens=_optional_int(state.get("external_reasoner_max_tokens")),
        temperature=_optional_float(state.get("external_reasoner_temperature")),
        top_p=_optional_float(state.get("external_reasoner_top_p")),
        timeout_s=_optional_float(state.get("external_reasoner_timeout_s")),
        ngl=_optional_int(state.get("external_reasoner_ngl")),
        ctx_size=_optional_int(state.get("external_reasoner_ctx_size")),
        batch_size=_optional_int(state.get("external_reasoner_batch_size")),
        ubatch_size=_optional_int(state.get("external_reasoner_ubatch_size")),
        threads=_optional_int(state.get("external_reasoner_threads")),
        threads_batch=_optional_int(state.get("external_reasoner_threads_batch")),
        mlock=_optional_bool(state.get("external_reasoner_mlock")),
        allow_cpu_fallback=bool(state.get("external_reasoner_allow_cpu_fallback", False)),
    )
    latency = float(result.get("latency_s") or 0.0)
    prompt_tps = result.get("prompt_tps")
    prompt_tps_float = float(prompt_tps) if isinstance(prompt_tps, (int, float)) else 0.0
    generation_tps = result.get("generation_tps")
    generation_tps_float = float(generation_tps) if isinstance(generation_tps, (int, float)) else 0.0

    if not result.get("ok"):
        return _result(
            status="skip" if result.get("error_type") == "configuration_error" else "error",
            state_delta={
                "external_reasoner_used": False,
                "external_reasoner_ok": False,
                "external_reasoner_schema_validated": False,
                "external_reasoner_structured_output_mode": result.get("structured_output_mode"),
                "external_reasoner_grammar_used": bool(result.get("grammar_used")),
                "external_reasoner_json_schema_used": bool(result.get("json_schema_used")),
                "external_reasoner_error_type": result.get("error_type"),
                "external_reasoner_error_message": result.get("error_message"),
                "external_reasoner_latency_s": latency,
                "external_reasoner_prompt_tps": prompt_tps_float,
                "external_reasoner_generation_tps": generation_tps_float,
                "external_reasoner_prompt_style": prompt_style,
                "external_reasoner_prompt_bytes": prompt_bytes,
            },
            confidence=0.0,
            cost=max(latency, 0.1),
            artifacts={"backend": result.get("backend"), "prompt_bytes": prompt_bytes},
            failure_mode=str(result.get("error_type") or "external_reasoner_error"),
            extras={
                "ok": False,
                "error_type": result.get("error_type"),
                "error_message": result.get("error_message"),
            },
        )

    raw_output = str(result.get("output_text") or result.get("stdout") or "")
    allowed = _allowed_interventions(state)
    try:
        parsed = parse_external_reasoner_payload(raw_output, allowed_interventions=allowed)
    except ExternalReasonerParseError as exc:
        return _result(
            status="error",
            state_delta={
                "external_reasoner_used": True,
                "external_reasoner_ok": False,
                "external_reasoner_schema_validated": False,
                "external_reasoner_structured_output_mode": result.get("structured_output_mode"),
                "external_reasoner_grammar_used": bool(result.get("grammar_used")),
                "external_reasoner_json_schema_used": bool(result.get("json_schema_used")),
                "external_reasoner_error_type": exc.code,
                "external_reasoner_error_message": str(exc),
                "external_reasoner_raw_output_excerpt": _strip_think_blocks(raw_output)[:_RAW_EXCERPT_LIMIT],
                "external_reasoner_latency_s": latency,
                "external_reasoner_prompt_tps": prompt_tps_float,
                "external_reasoner_generation_tps": generation_tps_float,
                "external_reasoner_prompt_style": prompt_style,
                "external_reasoner_prompt_bytes": prompt_bytes,
            },
            confidence=0.0,
            cost=max(latency, 0.1),
            artifacts={
                "backend": result.get("backend"),
                "structured_output_mode": result.get("structured_output_mode"),
                "grammar_used": bool(result.get("grammar_used")),
                "json_schema_used": bool(result.get("json_schema_used")),
                "schema_validated": False,
                "prompt_tps": result.get("prompt_tps"),
                "prompt_bytes": prompt_bytes,
            },
            failure_mode=exc.code,
            extras={"ok": False, "error_type": exc.code, "error_message": str(exc)},
        )

    contract_json = json.dumps(parsed, ensure_ascii=True, sort_keys=True)
    state_delta = {
        "external_reasoner_used": True,
        "external_reasoner_ok": True,
        "external_reasoner_schema_validated": True,
        "external_reasoner_structured_output_mode": result.get("structured_output_mode"),
        "external_reasoner_grammar_used": bool(result.get("grammar_used")),
        "external_reasoner_json_schema_used": bool(result.get("json_schema_used")),
        "external_reasoner_claim": parsed["claim"],
        "external_reasoner_reasoning_summary": parsed["reasoning_summary"],
        "external_reasoner_candidate_hypotheses": parsed["candidate_hypotheses"],
        "external_reasoner_causal_assumptions": parsed["causal_assumptions"],
        "external_reasoner_counterfactual_checks": parsed["counterfactual_checks"],
        "external_reasoner_confidence_proxy": parsed["confidence_proxy"],
        "external_reasoner_recommended_intervention": parsed["recommended_intervention"],
        "external_reasoner_raw_output_excerpt": contract_json[:_RAW_EXCERPT_LIMIT],
        "external_reasoner_latency_s": latency,
        "external_reasoner_prompt_tps": prompt_tps_float,
        "external_reasoner_generation_tps": generation_tps_float,
        "external_reasoner_prompt_style": prompt_style,
        "external_reasoner_prompt_bytes": prompt_bytes,
    }
    return _result(
        status="ok",
        state_delta=state_delta,
        confidence=parsed["confidence_proxy"],
        cost=max(latency, 0.1),
        artifacts={
            "backend": result.get("backend"),
            "prompt_tps": result.get("prompt_tps"),
            "generation_tps": result.get("generation_tps"),
            "structured_output_mode": result.get("structured_output_mode"),
            "grammar_used": bool(result.get("grammar_used")),
            "json_schema_used": bool(result.get("json_schema_used")),
            "schema_validated": True,
            "prompt_bytes": prompt_bytes,
        },
        extras={
            "ok": True,
            "schema_validated": True,
            "structured_output_mode": result.get("structured_output_mode"),
            "claim": parsed["claim"],
            "reasoning_summary": parsed["reasoning_summary"],
            "candidate_hypotheses": parsed["candidate_hypotheses"],
            "causal_assumptions": parsed["causal_assumptions"],
            "counterfactual_checks": parsed["counterfactual_checks"],
            "confidence_proxy": parsed["confidence_proxy"],
            "raw_output_excerpt": contract_json[:_RAW_EXCERPT_LIMIT],
            "latency_s": latency,
            "prompt_tps": prompt_tps_float,
            "generation_tps": generation_tps_float,
        },
    )
