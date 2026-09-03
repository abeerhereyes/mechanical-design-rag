from eval.run_multicourse_eval import evaluate


def test_multicourse_retrieval_has_no_course_leakage():
    results = evaluate(top_k=3)
    assert set(results) == {"aerodynamics", "qrm"}
    assert all(metrics["course_leaks"] == 0 for metrics in results.values())
    assert all(metrics["hit@3"] >= 0.5 for metrics in results.values())
