import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

import wandb
from importlib import import_module
from config import Config



def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str, default=Config.data_dir)
                        # default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default=Config.model_dir)
                        # default=os.environ.get('SM_MODEL_DIR','trained_models'))

    parser.add_argument('--device', default=Config.device)
                        # default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=Config.num_workers)

    parser.add_argument('--image_size', type=int, default=Config.image_size)
    parser.add_argument('--input_size', type=int, default=Config.input_size)
    parser.add_argument('--batch_size', type=int, default=Config.batch_size)
    parser.add_argument('--learning_rate', type=float, default=Config.learning_rate)
    parser.add_argument('--max_epoch', type=int, default=Config.max_epoch)
    parser.add_argument('--save_interval', type=int, default=Config.save_interval)

    parser.add_argument('--optimizer', type=str, default=Config.optimizer)
    parser.add_argument('--expr_name', type=str, default=Config.expr_name)
    parser.add_argument('--resume_from', type=str, default=Config.resume_from)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args



def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, optimizer, expr_name, resume_from):
    dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start_epoch=0
    model = EAST()
    model.to(device)
    opt_module = getattr(import_module("torch.optim"), optimizer)  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    if resume_from:         # 이어서 학습
        model_data = torch.load(resume_from)
        model.load_state_dict(model_data['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(model_data['optimizer_state_dict'])
        start_epoch= model_data['epoch']

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)


    min_loss = 10.
    model.train()
    er_cnt=0
    for epoch in range(start_epoch, max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                extra_info = None
                output = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                if isinstance(output, tuple):
                    loss, extra_info = output
                else:
                    loss = output
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = dict()
                if extra_info:
                    val_dict = {
                        'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                        'IoU loss': extra_info['iou_loss']
                    }
                else:
                    val_dict = {
                        'Cls loss': None, 'Angle loss': None,
                        'IoU loss': None
                    }
                wandb.log(val_dict)
                pbar.set_postfix(val_dict)
        scheduler.step()
        wandb.log({"Mean loss": epoch_loss / num_batches, "epoch": epoch})
        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if isinstance(epoch_loss / num_batches, float):
            if min_loss > epoch_loss / num_batches:
                er_cnt = 0

                min_loss = epoch_loss / num_batches
                print('Best Mean loss: {:.4f}'.format(min_loss))
                if not osp.exists(model_dir):
                    os.makedirs(model_dir)

                ckpt_fpath = osp.join(model_dir, 'best_mean_loss.pth')
                torch.save({
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict()},
                    ckpt_fpath)
            else:
                er_cnt += 1
                if er_cnt >= 5:
                    print(f'Early Stopping at epoch{epoch+1}!')
                    break

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict()},
                ckpt_fpath)
        

def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    wandb.init(project="data-annotation", name=args.expr_name, entity="level2-cv-09")
    wandb.config.update(args)
    main(args)
