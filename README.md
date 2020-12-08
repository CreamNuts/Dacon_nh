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

## 사용 방법
```shell
$ python3 main.py -h
```

```console
usage: main.py [-h] --model {kobert,distilkobert} [--ratio 0.8]
               [--data_dir ./data/news_train.csv] [--load None]
               [--save ./Checkpoint.pt] [--gpu 0] [--batchsize 32] [--lr 5e-05]
               [--epoch 5]
               {train,val,test}

positional arguments:
  {train,val,test}      train: Use total dataset, val: Hold-out of 0.8 ratio,
                        test: Make submission

optional arguments:
  -h, --help            show this help message and exit
  --model {kobert,distilkobert}
                        Select model
  --ratio 0.8           Hold out ratio in val mode
  --data_dir ./data/news_train.csv
                        Dataset Directory
  --load None           To continue your training, put your checkpoint dir
  --save ./Checkpoint.pt
                        Save directory
  --gpu 0               GPU number to use. If None, use CPU
  --batchsize 32        Train batch size
  --lr 5e-05
  --epoch 5
```

* test가 submission.csv를 만드는 것이므로 data_dir를 ./data/news_test.csv로 바꿔야함