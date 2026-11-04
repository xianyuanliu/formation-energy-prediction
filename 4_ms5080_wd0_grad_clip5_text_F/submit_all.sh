#!/bin/bash

# YAML 파일들이 들어있는 디렉토리 경로
CONFIG_DIR="examples/configs"

echo "===== Starting Job Submissions ====="

# 이전 작업의 Job ID를 저장할 변수 (초기에는 비워둠)
PREV_JOB_ID=""

# 해당 디렉토리의 모든 .yaml 파일에 대해 반복
for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do
    FILE_NAME=$(basename "$CONFIG_FILE")
    
    # 파일명이 exclusion pattern인 경우 건너뛰기
    if [[ "$FILE_NAME" == a* || "$FILE_NAME" == a* || "$FILE_NAME" == a* || "$FILE_NAME" == a* ]]; then
        echo "Skipping: $FILE_NAME (Matched exclusion pattern)"
        continue
    fi

    BASE_NAME=$(basename "$CONFIG_FILE" .yaml)
    
    echo "Submitting job for: $BASE_NAME"

    if [ -z "$PREV_JOB_ID" ]; then
        # 이전 작업이 없으므로(첫 번째 작업) 바로 제출
        # --parsable 옵션은 제출 후 긴 메세지 대신 "Job ID 숫자"만 딱 출력해 줍니다.
        PREV_JOB_ID=$(sbatch --parsable run_slurm_gpu.sh "$CONFIG_FILE" "$BASE_NAME")
        echo "  -> Started first job (ID: $PREV_JOB_ID)"
    else
        # 이전 작업 ID가 있다면, 그 작업이 끝난 뒤에 시작하도록 dependency 걸어서 제출
        PREV_JOB_ID=$(sbatch --parsable --dependency=afterany:$PREV_JOB_ID run_slurm_gpu.sh "$CONFIG_FILE" "$BASE_NAME")
        echo "  -> Queued job (ID: $PREV_JOB_ID) with dependency on previous job"
    fi

done

echo "===== All targeted jobs submitted (Sequentially linked)! ====="

