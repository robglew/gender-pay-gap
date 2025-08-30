import pandas as pd
from pathlib import Path
from src.train import train

def test_train_tmp(tmp_path: Path):
    csv = tmp_path / "pay.csv"
    df = pd.DataFrame({
        "jobtitle":["A","B","A","B"], "gender":["Male","Female","Male","Female"],
        "age":[30,40,35,29], "perfeval":[4,3,5,3],
        "education":["B","M","B","M"], "dept":["X","Y","X","Y"], "seniority":["S1","S2","S1","S2"],
        "basepay":[50000,60000,52000,61000], "bonus":[5000,6000,5200,5900]
    })
    df.to_csv(csv, index=False)
    out_dir = tmp_path / "artifacts"
    metrics = train(str(csv), str(out_dir))
    assert "mae" in metrics and "rmse" in metrics
    assert (out_dir / "model.pkl").exists()
    assert (out_dir / "metrics.json").exists()
