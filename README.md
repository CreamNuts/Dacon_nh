# Dacon: [AI야, 진짜 뉴스를 찾아줘!](https://dacon.io/competitions/official/235658/overview/)
## 진행 과정
* 데이터셋 구조 : test와 sample_submission의 샘플 순서가 같으므로, test시 suffle않고 추론후 그대로 sample_submission에 붙여넣음
    ```shell
    data/
        news_test.csv
        news_train.csv
        sample_submission.csv
    ```
* Kobert, Distilkobert로 실험함
* 3 Epoch, holdout ratio 0.75 
    * kobert : 0.9845 
    * distilkobert : 0.9822
* .py로 변환 후 시드 고정했을때 Kobert가 제대로 학습되지 않음. 특정 시드에서는 학슴이 안되고 특정 시드는 2 epoch까지 성능이 향상되다가 이후 trainloss가 크게 높아지면서 val acc도 함께 낮아짐. 원인 불명
* Distilkobert 7 epoch이 public 0.9877로 현재 최고 성능. Kobert의 학습 실패 원인을 찾기위해 추가적으로 다른 모델은 돌려보지 않음.
  > Bert-Multiligual에서도 비슷한 현상이 발견됨. Train loss도 같이 높아지므로 Overfit은 아닌것 같은데... 면밀한 loss 확인을 위해 배치당 loss를 시각화하고 validation 주기를 줄여서 확인할 예정
* **대회 끝! 자연어 맛보기로 적절했음. 예선은 통과했으나 추가 개선점이 떠오르지 않아 포기...**


## 사용 방법
```shell
$ python3 main.py -h
```

```console
usage: main.py [-h] --model
               {bert,kobert,distilkobert,distilbert,smallkoelectra,albert}
               [--ratio 0.8] [--data_dir ./data/news_train.csv] [--load None]
               [--save ./checkpoint] [--gpu 0] [--tensorboard True]
               [--batchsize 32] [--lr 5e-05] [--epoch 5]
               {train,val,test}

positional arguments:
  {train,val,test}      train: Use total dataset, val: Hold-out of 0.8 ratio,
                        test: Make submission

optional arguments:
  -h, --help            show this help message and exit
  --model {bert,kobert,distilkobert,distilbert,smallkoelectra,albert}
                        Select model
  --ratio 0.8           Hold out ratio in val mode
  --data_dir ./data/news_train.csv
                        Dataset Directory
  --load None           To continue your training, put your checkpoint dir
  --save ./checkpoint   Save directory name
  --gpu 0               GPU number to use. If None, use CPU
  --tensorboard True    If True, use Tensorboard
  --batchsize 32        Train batch size
  --lr 5e-05
  --epoch 5
```

* ~~test가 submission.csv를 만드는 것이므로 data_dir를 ./data/news_test.csv로 바꿔야함~~
  > test의 경우 data_dir 수정 안해도 미리 선언한 test 경로로 가도록 수정
* Tensorboard 추가
* Albert, smallkoelectra 추가
* save_dir을 디렉토리 이름 받는 것으로 수정