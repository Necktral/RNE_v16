from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler


def test_meta_scheduler_primary_sequence_trace():
    scheduler = MetaScheduler()
    result = scheduler.run({"episode": "t0"})
    sequence = [step["family"] for step in result["trace"]]
    assert sequence == ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]
    assert result["meta_family"] == "META"

