# EIS 열화 진단 — Zhang et al. (2020) Python 재현과 확장 분석

주논문: Y. Zhang et al., *Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning*, Nat. Commun. 11, 1706 (2020).
데이터: Zenodo 10.5281/zenodo.3633835 (state V EIS 12개, 용량 12개). 저자 MATLAB 코드: github.com/YunweiZhang/ML-identify-battery-degradation.

## 실행

```
pip install -r requirements.txt
# data/eis/ 에 EIS_state_V_*.txt, data/cap/ 에 Data_Capacity_*.txt
python run_all.py          # 또는 python src/repro.py 처럼 하나씩
```
CPU 2코어 기준 전체 약 30분. 그림과 수치는 `fig/`에 저장됩니다.

## 파일

| 파일 | 내용 |
|---|---|
| `src/eis_data.py` | raw txt 읽기, EIS 120 feature(Re 60 + −Im 60)와 방전 용량 매칭, EoL·RUL 라벨 |
| `src/models.py` | 논문 GPR 3종(SE-iso, LIN, 지표) scikit-learn 구현 |
| `src/gp_jax.py` | ARD GPR(JAX, 정확한 기울기) + 여러 시작점 재시작 |
| `src/cnn_jax.py` | 1D CNN과 gradient saliency |
| `src/repro.py` | 논문 Fig. 1–4 재현 + ARD 안정성(Fig. S1) |
| `src/ext1_cnn_saliency.py` | 확장 1 — 1D CNN + saliency map (Liu et al., JACS 2024) |
| `src/ext2_intercell.py` | 확장 2 — inter-cell 학습 (Zhang et al., Nat. Mach. Intell. 2025, BatLiNet) |
| `src/ext3_closed_loop.py` | 확장 3 — 불확실성 기반 측정 선택, 조기 수명 예측 (Attia et al., Nature 2020) |
| `src/ext4_benchmark.py` | 확장 4 — 모델 계열 벤치마크, 주파수 기여도 합의 (Ng et al., Nat. Mach. Intell. 2020 리뷰) |

Li et al. (ACS Energy Lett. 2026, 그래프 분석)은 3D X-선 미세구조 이미지용 방법이라 1D EIS 스펙트럼에는 적용하지 않았습니다.

## 전처리에서 확인한 사실

- **25C02 ↔ 25C03 용량 파일 짝**: 저자 MATLAB 행렬은 EIS 25C02를 용량 파일 25C03과, EIS 25C03을 용량 파일 25C02와 짝짓습니다(행 단위로 일치 확인). 같은 짝을 따랐습니다.
- **공개 raw 용량 파일이 일부 짧음**: 25C02·25C03·25C04·25C08 용량 파일은 EIS보다 일찍 끝납니다(25C08: EIS 86사이클, 용량 37사이클). 저자 행렬에는 그 뒤 값이 더 있어 학습 행 수가 1,358 → 1,216으로 줄었습니다. 용량이 있는 행만 썼습니다.
- **EIS k번째 ↔ 용량 k−1번째**, 전지 사이클 = 2k. 59점 스펙트럼은 log f에서 보간.
- **EoL** = 처음으로 초기 용량의 80% 미만이 되는 지점. 학습 셀은 저자 RUL.txt와 일치하지만, 시험 셀은 논문 값과 0–10사이클 차이(35C02: 논문 252, 여기 262).

## 재현 결과 (R²)

| 그림 | 셀 | 논문 | Python |
|---|---|---|---|
| Fig. 1a 용량 | 25C05 | 0.88 | 0.84 (mAh) |
| Fig. 2 RUL, 25 °C 모델 | 25C05 / 06 / 07 / 08 | 0.96 / 0.73 / 0.68 / 0.81 | 0.96 / 0.04 / −1.18 / −3.76 |
| Fig. 3 용량, 다중 온도 | 35C02 / 45C02 | 0.81 / 0.72 | 0.92 / 0.77 (정규화 곡선), 0.31 / 0.17 (mAh) |
| Fig. 4 RUL, 다중 온도 | 25C05 / 35C02 / 45C02 | 0.87 / 0.75 / 0.92 | 0.93 / 0.89 / 0.91 |

- 곡선 모양(45C02 초반의 볼록한 예측 포함)은 논문과 같습니다. 논문이 R²를 mAh로 냈는지 정규화 곡선으로 냈는지 명시하지 않아 둘 다 표시했습니다.
- 25C06–08의 RUL이 낮은 이유: 수명이 30사이클 남짓으로 점이 15–17개뿐이고, 25C08은 용량 파일이 짧아 EoL이 논문(38)과 다릅니다(32).
- **ARD는 시작점에 따라 고르는 주파수가 달라집니다**(Fig. S1). 35C01에서 논문의 91번째(−Im 17.8 Hz)는 7번 중 2번만 최상위였고, 45C01은 바로 옆인 90번째(22.5 Hz)가 나왔습니다. 인접 주파수끼리 거의 같은 정보를 담고 있어서입니다. "17.8 Hz 하나가 핵심"보다는 "수십 Hz 이하 −Im 영역"으로 읽는 것이 안전합니다.

## 확장 결과 요약

| 기법 (출처) | 답하려는 질문 | 이 데이터에서 나온 것 | 한계 |
|---|---|---|---|
| 1D CNN + saliency (Liu 2024) | 모델이 스펙트럼의 어디를 보고 판단하나? | 용량 R²: 35C02 0.62, 45C02 0.86 (논문 GPR 0.31, 0.17). saliency가 −Im 0.1–0.2 Hz에 집중 — Nyquist에서 반원이 확산 꼬리로 꺾이는 지점. 이 대역 −Im 평균 하나로 12개 셀 SoH와 r = −0.76 (고주파 저항 r = 0.21) | 상관이지 인과가 아님. 25 °C 일부 셀은 GPR보다 나쁨 |
| Inter-cell 학습 (BatLiNet 2025) | 조건이 다른 셀로 옮겨가도 버티나? | 같은 CNN이 셀 간 차이로 학습하자 평균 RUL RMSE 34 → 25사이클. 6개 중 4개 셀(25C05·06·07, 35C02)에서 논문 LIN-GPR보다 우수 | 평균은 LIN-GPR(23)이 근소 우위. 25C08·45C02에서 큼 |
| 불확실성 기반 선택 (Attia 2020) | 용량 측정을 몇 번이나 해야 하나? | 불확실성 순서가 무작위보다 뚜렷이 낫지 않음. 대신 20–30개 스펙트럼만으로 RMSE ≈ 2.6 mAh — 1,216개 전부(4.0)보다 작음 | 병목은 라벨 수가 아니라 셀 간 일반화 |
| 조기 수명 예측 (Attia/Severson) | 첫 20사이클로 EoL을 맞히나? | 11개 셀 leave-one-out MAPE 105 % (평균값 기준선 181 %). 45C01은 크게 빗나감 | 셀 11개로는 결론 불가, 참고용 |
| 모델 벤치마크 (Ng 2020 리뷰) | 어떤 모델 계열이 맞나? | 평균 RMSE(mAh): Elastic net 1.90 < GB 2.52 ≈ RF 2.54 < MLP 2.74 < CNN 2.89 < GPR-iso 3.74 < k-NN 4.04 < SVR 4.28. 기여도(< 1 Hz 몫): GPR 순열 0.62, RF 0.75, ARD 0.48, CNN 0.36(최고점은 −Im 0.13 Hz), elastic net 0.22(1–100 Hz가 0.54) | 단일 split, 하이퍼파라미터 기본값 |

## 해석 시 주의

- 셀 12개, LCO/graphite 코인셀 한 종류, 충방전 속도 고정입니다. 여기서 나온 순위·대역은 이 데이터에 대한 것입니다.
- 같은 데이터에 여러 기법을 돌리면 우연히 좋은 결과를 고를 위험이 있습니다. 기법끼리 **합의하는 결과**(저주파 −Im)를 결론으로, 갈리는 결과(최상위 주파수 하나)는 가설로 다루세요.
- 연구실 셀(LMB, 파우치)에 쓰려면 같은 코드에 데이터만 바꿔 다시 학습해야 합니다(`eis_data.py`의 주파수 보간이 장비 간 주파수 차이를 맞춰 줍니다).
