# scripts/launch_training.py
import os
import boto3
import sagemaker
from botocore.exceptions import ClientError
from sagemaker.sklearn.estimator import SKLearn

# --- CONFIG: set these 3 values ---
REGION = "us-east-2"  # your quota region
ROLE_ARN = "arn:aws:iam::336544441652:role/SageMakerExecutionRoleNoBoundary"  # <-- replace with your role ARN
BUCKET = "gender-pay-gap-mlops-rg-us-east-2"          # <-- your S3 bucket in us-east-2
TRAIN_CHANNEL = "s3://gender-pay-gap-mlops-rg-us-east-2/gender-pay-gap/train/"             # data/data.csv uploaded here
# ----------------------------------

PREFERRED = "ml.m5.large"
FALLBACKS = ["ml.t3.medium", "ml.m5.large"]  # order matters; PREFERRED is attempted first

def launch(instance_type: str):
    sess = sagemaker.Session(boto3.session.Session(region_name=REGION))
    print(f"[INFO] Using region={sess.boto_region_name}, instance_type={instance_type}")
    est = SKLearn(
        entry_point="sagemaker/train_sagemaker.py",  # your training script
        source_dir=".",                               # repo root (so it can import src/*)
        role=ROLE_ARN,
        framework_version="1.2-1",
        py_version="py3",
        instance_type=instance_type,
        instance_count=1,
        base_job_name="gender-pay-gap-train",
        sagemaker_session=sess,
    )
    est.fit({"train": TRAIN_CHANNEL})
    print("[SUCCESS]", instance_type, "->", est.model_data)
    return est.model_data

if __name__ == "__main__":
    tried = []
    # Try preferred first, then fallbacks
    for itype in [PREFERRED] + [i for i in FALLBACKS if i != PREFERRED]:
        try:
            artifact = launch(itype)
            break
        except ClientError as e:
            tried.append(itype)
            msg = str(e)
            print(f"[WARN] Failed on {itype}: {msg}")
            # Quota / limit style errors that should trigger fallback
            quota_markers = (
                "ResourceLimitExceeded",
                "LimitExceededException",
                "Instances",  # sometimes in the message text
            )
            if any(m in msg for m in quota_markers):
                print("[INFO] Looks like a quota/limit block; trying next instance type...")
                continue
            # Region or role mistakes shouldn’t be retried with a new instance type
            raise

    else:
        raise SystemExit(f"[ERROR] All instance types failed: {tried}")
