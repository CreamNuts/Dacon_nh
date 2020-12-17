import os, torch, argparse, random
import numpy as np
import pandas as pd
from utils import train, val, test, save
from dataset import create_loader
from tqdm import tqdm, trange
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, ElectraForSequenceClassification, AlbertForSequenceClassification, AdamW
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'val', 'test'], help='train: Use total dataset, val: Hold-out of 0.8 ratio, test: Make submission')
    parser.add_argument('--model', required=True, choices=['bert', 'kobert', 'distilkobert', 'distilbert', 'smallkoelectra', 'albert'], help='Select model')
    parser.add_argument('--ratio', type=float, metavar=0.8, default=0.8, help='Hold out ratio in val mode')
    parser.add_argument('--data_dir', metavar='./data/news_train.csv', default='./data/news_train.csv', help="Dataset Directory")
    parser.add_argument('--load', metavar='None', default=None, help='To continue your training, put your checkpoint dir')
    parser.add_argument('--save', metavar='./checkpoint', default='./checkpoint', help='Save directory name')
    parser.add_argument('--gpu', metavar=0, default=0, help='GPU number to use. If None, use CPU')
    parser.add_argument('--tensorboard', metavar=True, default=True, help='If True, use Tensorboard')
    parser.add_argument('--batchsize', type=int, metavar=32, default=32, help='Train batch size')
    parser.add_argument('--lr', type=float, metavar=5e-5, default=5e-5)
    parser.add_argument('--epoch', type=int, metavar=5, default=5)
    args = parser.parse_args()

    torch.manual_seed(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(777)
    random.seed(777)


    if args.gpu == None:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
    print('use: ', device)
    
    if args.model == 'kobert':
        model = BertForSequenceClassification.from_pretrained(f'monologg/{args.model}').to(device)
    elif args.model == 'distilkobert':
        model = DistilBertForSequenceClassification.from_pretrained(f'monologg/{args.model}').to(device)
    elif args.model == 'bert':
        model = BertForSequenceClassification.from_pretrained(f'bert-base-multilingual-cased').to(device)
    elif args.model == 'distilbert':
        model = DistilBertForSequenceClassification.from_pretrained(f'distilbert-base-multilingual-cased').to(device)        
    elif args.model == 'smallkoelectra':
        model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-v3-discriminator").to(device)
    elif args.model == 'albert':
        model = AlbertForSequenceClassification.from_pretrained("albert-base-v2").to(device)

    if args.load is not None:
        model.load_state_dict(torch.load(args.load))

    if args.mode != 'test':
        if args.tensorboard is True:
            writer = SummaryWriter(f'runs/{args.model}_{args.mode}_{args.lr}_{args.batchsize}')
        else: writer = None
        train_loader, val_loader = create_loader(args.data_dir, args.model, args.mode, batch_size=args.batchsize, ratio=args.ratio)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        for epoch in range(args.epoch):
            train(epoch, train_loader, val_loader, optimizer, model, device, args.save, writer)

    else:
        args.data_dir = './data/news_test.csv'
        test_loader, _ = create_loader(args.data_dir, args.model, args.mode, batch_size=args.batchsize)
        info_list = test(test_loader, model, device)
        submission = pd.read_csv('./data/sample_submission.csv')
        submission['info'] = info_list.cpu()
        submission.to_csv('submission.csv', index = False)
        