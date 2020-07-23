# 작업중인 상황

- augmentation으로 데이터 보충 (+pitch 넣어서 2ch로???)


# 데이터

- x_trian_MFCC.pickle : (sr = 22050, dtype = float32)

- x_trian.pickle : (sr = 16000, dtype = int32)

- x_trian_sr_11025.pickle : (sr = 11025, dtype = float32)


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
- deep CNN
- batchnormalize
