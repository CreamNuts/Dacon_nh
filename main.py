import os, torch, argparse
import pandas as pd
from utils import train, val, test, save
from dataset import create_loader
from tqdm import tqdm, trange
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, AdamW

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'val', 'test'], help='train: Use total dataset, val: Hold-out of 0.8 ratio, test: Make submission')
    parser.add_argument('--model', required=True, choices=['kobert', 'distilkobert'], help='Select model')
    parser.add_argument('--ratio', type=float, metavar=0.8, default=0.8, help='Hold out ratio in val mode')
    parser.add_argument('--data_dir', metavar='./data/news_train.csv', default='./data/news_train.csv', help="Dataset Directory")
    parser.add_argument('--load', metavar='None', default=None, help='To continue your training, put your checkpoint dir')
    parser.add_argument('--save', metavar='./Checkpoint.pt', default='./Checkpoint.pt', help='Save directory')
    parser.add_argument('--gpu', metavar=0, default='0', help='GPU number to use. If None, use CPU')
    parser.add_argument('--batchsize', type=int, metavar=32, default=32, help='Train batch size')
    parser.add_argument('--lr', type=float, metavar=5e-5, default=5e-5)
    parser.add_argument('--epoch', type=int, metavar=5, default=5)
    args = parser.parse_args()

    if args.gpu == None:
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('use: ',device)
    
    if args.model == 'kobert':
        model = BertForSequenceClassification.from_pretrained(f'monologg/{args.model}').to(device)
    elif args.model == 'distilkobert':
        model = DistilBertForSequenceClassification.from_pretrained(f'monologg/{args.model}').to(device)
    if args.load is not None:
        model.load_state_dict(torch.load(args.load))

    if args.mode != 'test':
        train_loader, val_loader = create_loader(args.data_dir, args.mode, batch_size=args.batchsize, ratio=args.ratio)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        for epoch in range(args.epoch):
            train(epoch, train_loader, optimizer, model, device)
            if args.mode == 'val':
                val(val_loader, model, device)
            save(model, epoch, args.mode, args.save)

    else:
        test_loader, _ = create_loader(args.data_dir, args.mode, batch_size=args.batchsize)
        info_list = test(test_loader, model, device)
        submission = pd.read_csv('./data/sample_submission.csv')
        submission['info'] = info_list.cpu()
        submission.to_csv('submission.csv', index = False)
        