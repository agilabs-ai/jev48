from jev48.metrics import accuracy,brier,nll,expected_calibration_error,risk_coverage


def test_metrics_perfect():
    p=[[1,0],[0,1]]; y=[[1,0],[0,1]]
    assert accuracy(p,y)==1
    assert brier(p,y)==0
    assert nll(p,y) < 1e-9
    assert expected_calibration_error(p,y) < 1e-9


def test_risk_coverage_shape():
    p=[[.9,.1],[.55,.45]]; y=[[1,0],[0,1]]
    rows=risk_coverage(p,y,thresholds=(.8,))
    assert rows[0]["coverage"]==0.5
    assert rows[0]["error_rate"]==0
