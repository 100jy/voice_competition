# 참고

https://sike6054.github.io/blog/paper/fourth-post/

# 작업중인 상황

- augmentation으로 데이터 보충 (heavy noise 추가 해보기, 다른 건 별로...)
- inception-resnet 써보기
- feature spectogram으로 바꿔보기..(128x128 mel-spectogram 사용,sr = 16k로..)
- 1D conv와 ensemble(raw_wav_16k_float32)


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
- inception net : CNN을 wide하게 한 레이어에서 다양한 필터 적용시키는 방법, 수렴속도 좋음, 정확도 상승


##참고

I had no background in speech recognition at all, but thanks to many generous kernels/discussions and I could learned a lot during this competition. Especially thanks to @ttagu99, @Heng, @vinvinvin

At first I wasn't interested in this competition, but 1D conv approach looked really interesting so I just gave it a try. Here's my initial approach:

1. Used 1D conv net with 10 pooling layer, used kernel size 9, filter count 256 for first layer.  Almost similar to ttagu99's, but replaced GAP+GMP to GMP and just used single FC with no dropout.
2. Split train/val by person id, and train/predict on all 10 folds. 
3. Listened mis-predicted samples from validation set(around 2000?) and noticed some mislabeled samples and samples without any voice in it. Smart guy would find another algorithm to identify these, but I just listened to them. Eventually I identified 640 silences, 121 mislabels.
4. Concatenated all noise and identified silences in training set into single wav. And it is randomly sampled while augmentation.
5. Augmentation: 
    a. Time-shift augmentation: many samples are clipped at start or end, so I thought it's better not to cut these out. So I just randomly padded samples front &amp; back with random noise and increased PCM sample count to 20k. I didn't expect this augmentation help much, somehow it helped somewhat. 
    b. Noise augmentation: Added up to x.5 noise and it improved LB little bit. 
    c. Tried other augmentations like pitch, volume, speed, but they didn't help much or even harmed the performance.
With this approach, I got LB score of .87 and couldn't increase the performance anymore with 1D conv. Tried some 2D Conv approach but didn't work well.

Later I formed a team with @Ildoo Kim who used high resolution mel spectrograms + VGG like network and had similar score as mine. We got immediate boost after merging of my augmentation and Ildoo's model. After some more fiddling of models, we got little bit of improvements. But we're stuck around .88 with single model, .893 with 5 model ensembles for a while.

I concluded that the model is large enough, so I worked more on data and found that adding heavy noise augmentation while keeping noise vs signal ratio doesn't exceed 2(yes, noise can be twice louder than voice) boost the score a lot. Just with this augmentation, same model (resnet-like net) got public LB score .898

Unfortunately we found this 2 days just before deadline and didn't have much time and submissions to experiment more. So we just trained a few more models and ensembled blindly even without checking individual scores.

One interesting thing was my original 1D model didn't work well after adding heavy noise. So I dropped it altogether. But later I found that 1D model also can get better score also once I add more capacity to the network.
