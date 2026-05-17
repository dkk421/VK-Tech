import json
import subprocess
from typing import Any


REMOTE_HOST = "team6@10.100.0.242"
REMOTE_PROJECT_DIR = "/home/team6/project/VK-Tech"


def run_remote_analysis(exam_row_id: int) -> dict[str, Any]:
    remote_command = (
        f"cd {REMOTE_PROJECT_DIR} && "
        f"python xgboost_agent.py "
        f"--train train.csv "
        f"--input test.csv "
        f"--exam-row-id {exam_row_id}"
    )

    result = subprocess.run(
        ["ssh", REMOTE_HOST, remote_command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)

    return json.loads(result.stdout)
