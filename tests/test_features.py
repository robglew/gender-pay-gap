import pandas as pd
from src.features import compute_targets, build_preprocessor
from src.schema import NUM_COLS, CAT_COLS

def test_compute_targets():
    df = pd.DataFrame({
        "jobtitle":["A"], "gender":["Male"], "age":[30], "perfeval":[4],
        "education":["B"], "dept":["X"], "seniority":["S1"],
        "basepay":[50000], "bonus":[5000]
    })
    df2 = compute_targets(df)
    assert "total_comp" in df2 and "log_total_comp" in df2
    assert df2.loc[0, "total_comp"] == 55000

def test_preprocessor_shapes():
    pre = build_preprocessor()
    df = pd.DataFrame({
        "jobtitle":["A","B"], "gender":["Male","Female"], "age":[30,40], "perfeval":[4,3],
        "education":["B","M"], "dept":["X","Y"], "seniority":["S1","S2"],
        "basepay":[50000,60000], "bonus":[5000,6000]
    })
    X = df[NUM_COLS + CAT_COLS]
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == 2
