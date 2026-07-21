from __future__ import annotations

from runtime.organism.teacher import Teacher


class _ExperienceRecorder:
    def __init__(self) -> None:
        self.records = []

    def record(self, record) -> None:
        self.records.append(record)


class _StorageRecorder:
    def __init__(self) -> None:
        self.events = []

    def append_event(self, **event) -> None:
        self.events.append(event)


class _InvalidSemanticClient:
    def __init__(self) -> None:
        self.last_prompt = ""

    def generate(self, *args, **kwargs):
        self.last_prompt = str(args[0])
        return {
            "ok": True,
            "output_text": '{"avoid":"free prose","prefer":"also prose","lesson":"LessoN"}',
            "latency_s": 6.54,
        }


def test_teacher_preference_is_proposal_not_fabricated_certified_success() -> None:
    experience = _ExperienceRecorder()
    storage = _StorageRecorder()
    teacher = Teacher(storage=storage, experience=experience)

    teacher._persist_lesson(
        {
            "lesson_id": "lesson-safe",
            "organism_id": "organism",
            "situation_key": "situation",
            "scenario": "thermal_homeostasis",
            "regime": "alarm",
            "avoid": "deactivate_cooling",
            "prefer": "activate_cooling",
            "lesson": "evitar repetir el golpe",
            "from_severity": 0.8,
        }
    )

    assert len(experience.records) == 1
    scar = experience.records[0]
    assert scar.action == "teacher_scar"
    assert scar.verdict == "teacher_evidence"
    assert scar.metadata["prefer_proposal"] == "activate_cooling"
    assert all(record.action != "teacher_prefer" for record in experience.records)
    assert storage.events[0]["event_type"] == "experience.lesson"


def test_codex_lesson_enters_as_unproven_teacher_evidence() -> None:
    storage = _StorageRecorder()
    experience = _ExperienceRecorder()
    teacher = Teacher(storage=storage, experience=experience)

    lesson = teacher.register_external_lesson(
        teacher_source="codex_frontier",
        lesson={
            "organism_id": "org",
            "situation_key": "situation",
            "scenario": "scenario",
            "regime": "stress",
            "avoid": "bad",
            "prefer": "good",
            "lesson": "contrasta la alternativa antes de actuar",
            "from_severity": 0.8,
            "source_wound_episode_id": "wound-1",
        },
    )

    assert lesson["teacher_source"] == "codex_frontier"
    assert lesson["pedagogical_role"] == "external_teacher_curriculum_candidate"
    assert lesson["autonomous_teacher"] is False
    assert lesson["verification_authority"] == "domain_verifiers_and_paired_outcome"
    assert lesson["curriculum_promotion_authorized"] is False
    assert experience.records[0].verdict == "teacher_evidence"
    assert experience.records[0].metadata["pedagogical_role"] == (
        "external_teacher_curriculum_candidate"
    )
    assert experience.records[0].metadata["autonomous_teacher"] is False
    assert all(record.action != "teacher_prefer" for record in experience.records)
    assert storage.events[0]["payload"]["teacher_source"] == "codex_frontier"


def test_local_7b_semantic_failure_is_measured_and_bounded() -> None:
    storage = _StorageRecorder()
    experience = _ExperienceRecorder()
    teacher = Teacher(storage=storage, experience=experience)
    teacher._client = _InvalidSemanticClient()

    lesson = teacher._reflect_one(
        organism_id="org",
        wound={
            "episode_id": "wound-1",
            "situation_key": "situation",
            "scenario": "thermal_homeostasis",
            "regime": "alarm",
            "intervention": "deactivate_cooling",
            "severity": 0.8,
        },
        valid_interventions=["activate_cooling", "maintain_cooling", "deactivate_cooling"],
    )

    assert lesson is not None
    assert lesson["avoid"] == "deactivate_cooling"
    assert lesson["prefer"] == "activate_cooling"
    assert lesson["teacher_raw_semantic_valid"] is False
    assert lesson["teacher_latency_s"] == 6.54
    assert lesson["pedagogical_role"] == "supervised_student_reflection_proposer"
    assert lesson["autonomous_teacher"] is False
    assert lesson["verification_authority"] == "domain_verifiers_and_paired_outcome"
    assert "not an autonomous teacher" in teacher._client.last_prompt
    assert "Codex is the external teacher and curriculum source" in teacher._client.last_prompt
    assert set(lesson["teacher_repairs"]) == {
        "avoid_rebound_to_observed_wound",
        "prefer_rebound_to_valid_alternative",
        "lesson_replaced_by_bounded_fallback",
    }
