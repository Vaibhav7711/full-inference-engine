from engine.graphs import assess_graph_eligibility


def test_fixed_shape_static_cache_workload_is_graph_eligible() -> None:
    result = assess_graph_eligibility(fixed_batch_size=True, fixed_sequence_length=True, static_cache=True, dynamic_arrivals=False)
    assert result.eligible
    assert result.reasons == ()


def test_continuous_batching_is_not_a_single_cuda_graph_shape() -> None:
    result = assess_graph_eligibility(fixed_batch_size=False, fixed_sequence_length=False, static_cache=False, dynamic_arrivals=True)
    assert not result.eligible
    assert "batch size changes" in result.reasons
    assert "request arrivals/departures change control flow" in result.reasons
