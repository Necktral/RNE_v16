"""El maestro — la propia IA del organismo (7B) lo hace reflexionar sobre sus golpes.

*"Tiene su propia IA que es su maestra también."* Tras una herida, el 7B reflexiona:
dada esta situación y esta decisión que dolió, ¿cuál es la lección? Produce una lección
estructurada (qué evitar / qué preferir) que se graba como experiencia reforzada, de modo
que el sesgo de decisión (E3) la recoja: la próxima vez, el organismo elige distinto.

La reflexión es parte de su vida (se invoca proporcional al daño, off-hot-path). El 7B
corre en la GPU (RTX 2070 vía llama.cpp). Gated por ``RNFE_TEACHER`` (off ⇒ sin maestro).
"""

from __future__ import annotations

import json
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

        Devuelve las lecciones producidas (también las graba como experiencia
        reforzada para que el sesgo de decisión las use).
        """
        if not teacher_enabled():
            return []
        experiences = self.experience.recall(organism_id=organism_id, limit=200)
        wounds = [e for e in experiences if float(e.get("severity", 0.0)) >= WOUND_THRESHOLD and e.get("intervention")]
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
        avoid = str(parsed.get("avoid") or hurt_iv)
        prefer = str(parsed.get("prefer") or "")
        if prefer not in valid_interventions or prefer == avoid:
            prefer = alternatives[0] if alternatives else ""
        lesson_text = str(parsed.get("lesson") or "")[:280]
        lesson = {
            "organism_id": organism_id,
            "situation_key": situation,
            "scenario": scenario,
            "regime": regime,
            "avoid": avoid,
            "prefer": prefer,
            "lesson": lesson_text,
            "from_severity": severity,
        }
        self._persist_lesson(lesson)
        return lesson

    def _persist_lesson(self, lesson: Dict[str, Any]) -> None:
        """Graba la lección como experiencia REFORZADA (refuerza la cicatriz de lo que evitar,
        y una experiencia benigna de lo que preferir) para que el sesgo E3 la recoja, y como evento."""
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
                action="teacher_scar", verdict="uncertified",
                metadata={"lesson": lesson.get("lesson"), "origin": "teacher"},
            ))
            if lesson.get("prefer"):
                self.experience.record(ExperienceRecord(
                    organism_id=lesson.get("organism_id", ""), run_id="teacher", episode_id=f"lesson-prefer-{situation}",
                    situation_key=situation, scenario=scenario, regime=regime,
                    intervention=str(lesson["prefer"]), severity=0.0, wound=False,
                    viability_margin=1.0, ioc=1.0, risk=0.0, reward=1.0,
                    action="teacher_prefer", verdict="certified",
                    metadata={"lesson": lesson.get("lesson"), "origin": "teacher"},
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
