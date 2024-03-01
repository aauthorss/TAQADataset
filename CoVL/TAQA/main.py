import os
import sys

sys.path.append('../')

import torch
import torch.nn as nn

from utils import *
from opts import *
from scipy import stats
from tqdm import tqdm
from dataset import VideoDataset
from models.i3d import InceptionI3d
from models.evaluator import Evaluator
from config import get_parser

from text_prompt import text_prompt
import numpy
from torch.nn.functional import cosine_similarity
from transformers import BertModel,BertTokenizer

class TextEncoding(nn.Module):
    def __init__(self,) :
        super(TextEncoding, self).__init__()
        self.bert = BertModel.from_pretrained("./bert/")
        self.tokenizer = BertTokenizer.from_pretrained('./bert/')

        self.layer1 = nn.Linear(768, 1024).cuda()


    def forward(self,text):
        output=self.bert(text)
        output = output[0]

        return self.layer1(output)
        
def get_models(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    i3d = InceptionI3d().cuda()
    i3d.load_state_dict(torch.load(i3d_pretrained_path))

    text_model = TextEncoding().cuda()

    if args.type == 'USDL-CoVL':
        evaluator = Evaluator(output_dim=output_dim['USDL-CoVL'], model_type='USDL-CoVL').cuda()
    else:
        evaluator = Evaluator(output_dim=output_dim['MUSDL-CoVL'], model_type='MUSDL-CoVL', num_judges=num_judges).cuda()

    if len(args.gpu.split(',')) > 1:
        i3d = nn.DataParallel(i3d)
        evaluator = nn.DataParallel(evaluator)
    return i3d,text_model, evaluator


def compute_score(model_type, probs, data):
    if model_type == 'USDL-CoVL':
        pred = probs.argmax(dim=-1) * (label_max / (output_dim['USDL-CoVL']-1))
    else:
        judge_scores_pred = torch.stack([prob.argmax(dim=-1) * judge_max / (output_dim['MUSDL-CoVL']-1)
                                         for prob in probs], dim=1).sort()[0]

        pred = torch.sum(judge_scores_pred[:, 0:5], dim=1) * data['difficulty'].cuda()
    return pred


def compute_loss(model_type, criterion, probs, data):
    if model_type == 'USDL-CoVL':
        loss = criterion(torch.log(probs), data['soft_label'].cuda())
    else:
        loss = sum([criterion(torch.log(probs[i]), data['soft_judge_scores'][:, i].cuda()) for i in range(num_judges)])
    return loss

def Contrastive_Loss(video_repr, text_repr, margin=1.0):
    similarity = nn.functional.cosine_similarity(video_repr, text_repr)

    loss = 0.5 * ((1 - similarity) ** 2) 
    loss = torch.clamp(loss, min=0.0, max=margin) 

    return loss.mean()

def get_dataloaders(args):
    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(VideoDataset('train', args),
                                                       batch_size=args.train_batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       worker_init_fn=worker_init_fn)

    dataloaders['test'] = torch.utils.data.DataLoader(VideoDataset('test', args),
                                                      batch_size=args.test_batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      worker_init_fn=worker_init_fn)
    return dataloaders


def main(dataloaders, i3d, text_model,evaluator, base_logger, args):
    print('=' * 40)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('=' * 40)

    criterion = nn.KLDivLoss()
    optimizer = torch.optim.Adam([*i3d.parameters()] + [*text_model.parameters()]+ [*evaluator.parameters()],
                                 lr=args.lr, weight_decay=args.weight_decay)

    epoch_best = 0
    rho_best = 0
    for epoch in range(args.num_epochs):
        log_and_print(base_logger, f'Epoch: {epoch}  Current Best: {rho_best} at epoch {epoch_best}')

        for split in ['train', 'test']:
            true_scores = []
            pred_scores = []

            if split == 'train':
                i3d.train()
                text_model.train()
                evaluator.train()
                torch.set_grad_enabled(True)
            else:
                i3d.eval()
                text_model.eval()
                evaluator.eval()
                torch.set_grad_enabled(False)

            for data,list_id in tqdm(dataloaders[split]):
                true_scores.extend(data['final_score'].numpy())
                videos = data['video'].cuda()
                videos.transpose_(1, 2)  # N, C, T, H, W

                batch_size, C, frames, H, W = videos.shape
                clip_feats = torch.empty(batch_size, 10, feature_dim).cuda()
                for i in range(9):
                    clip_feats[:, i] = i3d(videos[:, :, 10 * i:10 * i + 16, :, :]).squeeze(2)
                clip_feats[:, 9] = i3d(videos[:, :, -16:, :, :]).squeeze(2)

                final_scores = []
                final_scores.extend(data['final_score'].numpy())
                num_text_aug, text_dict = text_prompt(final_scores)
                text_id = numpy.random.randint(num_text_aug, size=len(list_id))
                list_id_id = [i for i, x in enumerate(list_id)]

                texts = torch.stack(
                    [text_dict[j][i, :] for i, j in zip(list_id_id, text_id)])  
                texts = texts.cuda()
                text_feats = text_model(texts).mean(1)  

         
                contrastive_loss = Contrastive_Loss(clip_feats.mean(1), text_feats)


                probs = evaluator(clip_feats.mean(1))
                preds = compute_score(args.type, probs, data)
                pred_scores.extend([i.item() for i in preds])

                if split == 'train':
                    loss =102*compute_loss(args.type, criterion, probs, data) + contrastive_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            rho, p = stats.spearmanr(pred_scores, true_scores)

            log_and_print(base_logger, f'{split} correlation: {rho}')

        if rho > rho_best:
            rho_best = rho
            epoch_best = epoch
            log_and_print(base_logger, '-----New best found!-----')
            if args.save:
                torch.save({'epoch': epoch,
                            'i3d': i3d.state_dict(),
                            'evaluator': evaluator.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'rho_best': rho_best}, f'ckpts/{args.type}.pt')


if __name__ == '__main__':

    args = get_parser().parse_args()

    if not os.path.exists('./exp'):
        os.mkdir('./exp')
    if not os.path.exists('./ckpts'):
        os.mkdir('./ckpts')

    init_seed(args)

    base_logger = get_logger(f'exp/{args.type}.log', args.log_info)
    i3d,text_model, evaluator = get_models(args)
    dataloaders = get_dataloaders(args)

    main(dataloaders, i3d, text_model,evaluator, base_logger, args)
