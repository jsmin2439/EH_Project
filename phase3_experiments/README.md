# Phase 3 Experiment Workspace

`run_phase3_experiments.py`는 3차 발표용 실험 설계를 그대로 재현할 수 있도록 구성된 실행 스크립트입니다.

## 폴더 구조
- `run_phase3_experiments.py` : 데이터 전략, 모델 구조, loss/scheduler, 해상도 비교 등 네 축별 실험을 한 번에 실행하는 진입점
- `se_resnet.py` : ResNet32에 SE(Channel Attention)를 삽입한 경량 모델 정의

## 실행 방법
기본 경로가 이미 `dataset/digit_data`와 연결되어 있으므로 아래 명령만으로 전체 실험을 순차 실행할 수 있습니다.

```bash
python phase3_experiments/run_phase3_experiments.py
```

### 자주 쓰는 옵션
- `--groups`: `data,model,loss,superres` 중 원하는 축만 선택 (`--groups data,loss` 또는 `--groups all`)
- `--device`: `cuda`, `cpu`, `auto` 중 선택 (`auto`는 GPU 사용 가능 시 자동 선택)
- `--epochs`, `--batch_size`, `--lr`, `--weight_decay`: 전체 실험 공통 하이퍼파라미터 override
- `--output_dir`: 실험 결과 저장 경로 (default: `phase3_runs`)
- `--small_*_list`, `--noisy_*_list`, `--hard_*_list`: 하드 샘플 정의 파일을 교체하고 싶을 때 사용

### 실행 예시
```bash
# 데이터/샘플링 축만 실행 (GPU 자동 선택)
python phase3_experiments/run_phase3_experiments.py --groups data

# 모델·손실축만 실행, 배치 크기 조정, CPU 강제
python phase3_experiments/run_phase3_experiments.py --groups model,loss --batch_size 48 --device cpu

# 해상도 + TTA 실험만 따로 실행하고 출력 경로 지정
python phase3_experiments/run_phase3_experiments.py --groups superres --output_dir runs_superres
```

### 입력 리스트 교체
스몰/노이즈/하드 샘플 리스트를 커스텀하려면 `dataset/digit_data`에 새로운 txt를 만든 뒤 다음과 같이 지정합니다.
```bash
python phase3_experiments/run_phase3_experiments.py \
  --small_train_list dataset/digit_data/my_small_train.txt \
  --small_valid_list dataset/digit_data/my_small_valid.txt
```

### 결과 확인
- `phase3_runs/<experiment_name>/train.log`: 각 epoch의 loss/acc 로그
- `phase3_runs/<experiment_name>/summary.json`: config, best val acc, 전체 history 요약
- `phase3_runs/<experiment_name>/eval_metrics.json`: Overall/Small/Noise/Hard 정확도, per-class accuracy, confusion matrix, 에러 사례
- `phase3_runs/<experiment_name>/best_model.pth`: 최고 성능 가중치

상위 폴더 `phase3_runs/phase3_summary.json`에는 실행된 모든 실험 결과가 리스트 형태로 기록되어 발표 자료용 표를 바로 구성할 수 있습니다.
