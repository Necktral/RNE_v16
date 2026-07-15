"""El maestro — la propia IA del organismo (7B) lo hace reflexionar sobre sus golpes.

*"Tiene su propia IA que es su maestra también."* Tras una herida, el 7B reflexiona:
dada esta situación y esta decisión que dolió, ¿cuál es la lección? Produce una lección
estructurada (qué evitar / qué preferir). La herida observada se refuerza; la
preferencia permanece como propuesta hasta que un outcome la contraste. El sesgo E3
puede evitar el daño previo sin fabricar que la alternativa ya fue exitosa.

La reflexión es parte de su vida (se invoca proporcional al daño, off-hot-path). El 7B
corre en la GPU (RTX 2070 vía llama.cpp). Gated por ``RNFE_TEACHER`` (off ⇒ sin maestro).
"""

from __future__ import annotations

import json
import hashlib
import os
from typing import Any, Dict, List, Optional

from runtime.storage import StorageFacade
from runtime.organism.experience import (
    ExperienceRecord,
    ExperienceStore,
    WOUND_THRESHOLD,
)

_TRUE = {"1", "true", "yes", "on"}


def teacher_enabled() -> bool:
    """True si el maestro (7B) reflexiona sobre los golpes del organismo."""
    return os.environ.get("RNFE_TEACHER", "").strip().lower() in _TRUE


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extrae el primer objeto JSON balanceado del texto (parseo defensivo del LLM)."""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


class Teacher:
    """Invoca el 7B para reflexionar sobre una herida y destilar una lección."""

    def __init__(self, *, storage: StorageFacade, experience: ExperienceStore):
        self.storage = storage
        self.experience = experience
        self._client = None  # lazy: el 7B es caro de construir

    def _get_client(self):
        if self._client is None:
            import dataclasses

            from runtime.reasoning.external_models.config import ExternalReasonerConfig
            from runtime.reasoning.external_models.llama_cpp_client import LlamaCppClient

            # El maestro reflexiona en formato propio {avoid,prefer,lesson}. Se le fija su
            # PROPIO schema (no el decision-shaped del razonador) para forzar JSON directo
            # — el 7B es un modelo de razonamiento que si no, "piensa en voz alta" sin cerrar.
            import os as _os

            cfg = ExternalReasonerConfig.from_env()
            lesson_schema = _os.path.join(
                _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                "reasoning", "external_models", "schemas", "experience_lesson.schema.json",
            )
            try:
                cfg = dataclasses.replace(
                    cfg, structured_output_mode="json_schema", json_schema_path=lesson_schema
                )
            except Exception:
                pass
            self._client = LlamaCppClient(cfg)
        return self._client

    def reflect(
        self,
        *,
        organism_id: str,
        valid_interventions: List[str],
        max_reflections: int = 1,
    ) -> List[Dict[str, Any]]:
        """Reflexiona sobre las peores heridas recientes → lecciones grabadas.

        Devuelve las lecciones producidas. Refuerza la herida observada y persiste
        la alternativa recomendada como propuesta no certificada.
        """
        if not teacher_enabled():
            return []
        experiences = self.experience.recall(organism_id=organism_id, limit=200)
        wounds = [
            e
            for e in experiences
            if float(e.get("severity", 0.0)) >= WOUND_THRESHOLD
            and e.get("intervention")
            and not (
                isinstance(e.get("metadata"), dict)
                and e["metadata"].get("origin") == "teacher"
            )
        ]
        if not wounds:
            return []
        # Las heridas más profundas primero (∝ daño): el maestro atiende lo que más dolió.
        wounds.sort(key=lambda e: float(e.get("severity", 0.0)), reverse=True)
        lessons: List[Dict[str, Any]] = []
        for wound in wounds[: max(1, int(max_reflections))]:
            lesson = self._reflect_one(
                organism_id=organism_id, wound=wound, valid_interventions=valid_interventions
            )
            if lesson is not None:
                lessons.append(lesson)
        return lessons

    def register_external_lesson(
        self,
        *,
        lesson: Dict[str, Any],
        teacher_source: str = "codex_frontier",
    ) -> Dict[str, Any]:
        """Registra una lección externa como hipótesis, nunca como éxito.

        Este es el puerto explícito para que Codex u otro docente produzca el mismo
        contrato que el 7B. Exige vínculo a una herida observada; la eficacia se
        decide después mediante ensayos curriculares pareados.
        """
        required = (
            "organism_id",
            "situation_key",
            "scenario",
            "regime",
            "avoid",
            "prefer",
            "lesson",
            "from_severity",
            "source_wound_episode_id",
        )
        missing = [name for name in required if lesson.get(name) in (None, "")]
        if missing:
            raise ValueError(f"external_lesson_missing:{','.join(missing)}")
        source = str(teacher_source).strip()
        if source not in {"codex_frontier", "local_7b", "human_mentor"}:
            raise ValueError("external_lesson_teacher_source_invalid")
        candidate = dict(lesson)
        candidate["teacher_source"] = source
        candidate.setdefault("teacher_model", source)
        candidate["lesson"] = str(candidate["lesson"])[:280]
        candidate["lesson_id"] = hashlib.sha256(
            json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._persist_lesson(candidate)
        return candidate

    def _reflect_one(
        self, *, organism_id: str, wound: Dict[str, Any], valid_interventions: List[str]
    ) -> Optional[Dict[str, Any]]:
        situation = str(wound.get("situation_key") or "")
        scenario = str(wound.get("scenario") or "")
        regime = str(wound.get("regime") or "")
        hurt_iv = str(wound.get("intervention") or "")
        severity = float(wound.get("severity", 0.0))
        alternatives = [iv for iv in valid_interventions if iv != hurt_iv]
        prompt = (
            "You are the inner teacher of a self-preserving cognitive organism. "
            "It just got hurt. Reflect briefly and give ONE lesson so it does not repeat the mistake.\n"
            f"Situation: scenario='{scenario}', regime='{regime}'.\n"
            f"It chose intervention '{hurt_iv}' and suffered a wound (severity {severity:.2f}, "
            f"viability_margin {float(wound.get('viability_margin',0)):.2f}, ioc {float(wound.get('ioc',0)):.2f}, "
            f"risk {float(wound.get('risk',0)):.2f}).\n"
            f"Valid interventions: {valid_interventions}.\n"
            "Respond ONLY with a JSON object: "
            '{"avoid": "<the intervention to avoid>", "prefer": "<a better intervention from the valid list>", '
            '"lesson": "<one short sentence>"}'
        )
        try:
            client = self._get_client()
            # Sin schema constraint (el schema global es decision-shaped); parseo defensivo.
            res = client.generate(prompt, backend="cuda", max_tokens=160)
        except Exception:
            return None
        if not res or not res.get("ok"):
            return None
        text = res.get("output_text") or res.get("stdout") or ""
        parsed = _extract_json(text)
        if not isinstance(parsed, dict):
            return None
        raw_avoid = str(parsed.get("avoid") or "")
        raw_prefer = str(parsed.get("prefer") or "")
        raw_lesson = str(parsed.get("lesson") or "")[:280]
        repairs: List[str] = []
        avoid = raw_avoid
        if avoid != hurt_iv:
            avoid = hurt_iv
            repairs.append("avoid_rebound_to_observed_wound")
        prefer = raw_prefer
        if prefer not in valid_interventions or prefer == avoid:
            prefer = alternatives[0] if alternatives else ""
            repairs.append("prefer_rebound_to_valid_alternative")
        lesson_text = raw_lesson
        if len(lesson_text.split()) < 4:
            lesson_text = (
                f"Avoid {avoid}; test {prefer} in the same situation and measure severity."
            )
            repairs.append("lesson_replaced_by_bounded_fallback")
        raw_semantic_valid = not repairs
        lesson = {
            "organism_id": organism_id,
            "situation_key": situation,
            "scenario": scenario,
            "regime": regime,
            "avoid": avoid,
            "prefer": prefer,
            "lesson": lesson_text,
            "from_severity": severity,
            "source_wound_episode_id": wound.get("episode_id"),
            "teacher_source": "local_7b",
            "teacher_model": "open-thoughts/OpenThinker3-7B",
            "teacher_latency_s": res.get("latency_s"),
            "teacher_prompt_tps": res.get("prompt_tps"),
            "teacher_generation_tps": res.get("generation_tps"),
            "teacher_raw_semantic_valid": raw_semantic_valid,
            "teacher_repairs": repairs,
        }
        lesson["lesson_id"] = hashlib.sha256(
            json.dumps(lesson, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._persist_lesson(lesson)
        return lesson

    def _persist_lesson(self, lesson: Dict[str, Any]) -> None:
        """Refuerza la herida observada y persiste la preferencia sólo como propuesta.

        La recomendación del 7B no se escribe como éxito certificado: todavía no fue
        ejecutada ni contrastada contra un outcome.
        """
        situation = lesson["situation_key"]
        scenario = lesson["scenario"]
        regime = lesson["regime"]
        # La cicatriz enseñada es fuerte (el maestro certifica el golpe): severidad alta.
        avoid_scar = min(1.0, 0.5 + 0.5 * float(lesson.get("from_severity", 0.6)))
        try:
            self.experience.record(ExperienceRecord(
                organism_id=lesson.get("organism_id", ""), run_id="teacher", episode_id=f"lesson-{situation}",
                situation_key=situation, scenario=scenario, regime=regime,
                intervention=str(lesson["avoid"]), severity=round(avoid_scar, 4), wound=True,
                viability_margin=0.0, ioc=0.0, risk=1.0, reward=-1.0,
                action="teacher_scar", verdict="teacher_evidence",
                metadata={
                    "lesson": lesson.get("lesson"),
                    "lesson_id": lesson.get("lesson_id"),
                    "prefer_proposal": lesson.get("prefer"),
                    "origin": "teacher",
                },
            ))
        except Exception:
            pass
        try:
            self.storage.append_event(
                event_type="experience.lesson",
                run_id="teacher",
                source="teacher",
                payload=lesson,
            )
        except Exception:
            pass
