# 작업중인 상황

- augmentation으로 데이터 보충 (location 이동시켜서 데이터 2배로)
- 


# 데이터

- x_trian_MFCC.pickle : (sr = 22050, dtype = float32, shape = 10^5 x 40 x 44)

- x_trian.pickle : (sr = 16000, dtype = int32, shape = 10^5 x 16000)

- x_trian_sr_11025.pickle : (sr = 11025, dtype = float32, shape = 10^5 x 11025)

- x_trian_mfcc_80.pickle : (sr = 22050, dtype = float32, shape = 10^5 x 80 x 80)


# voice_competition
중첩된 4개의 단어를 구분하는 task

## 음원에서 샘플링
어떤 feature로 접근할 것인가...

-> MFCC 메모리 적게 쓰고 분류가 잘 되는듯....

-> spectogram으로 t-sne 해본 결과 분류가능성 높아 보였음...

## EDA와 데이터 전처리
VAD하면 더 좋은 성능 보일 듯..(sparse함을 유발할듯...)

k-l divergence를 목적함수로 적당한가???

augmentation통해서 성능 향상 할 수 있을 듯..

## 모델링
효과 좋았던 것들...
- deep CNN : 필터개수 늘리고, 풀링 수 줄임..
- batchnormalize : 레이어 마다 배치 정규화하여 학습의 안정성 높임
- GELU : RELU의 smoother버젼, 모델 정확도 상승...
