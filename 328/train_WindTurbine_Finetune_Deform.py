import os
import argparse
import random
import time
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import kornia
import utils
from data.data_RGB import get_training_data, get_validation_data
from loss import losses
from warmup_scheduler import GradualWarmupScheduler
from models.MISCFilterNet_Deform import MISCKernelNet_Deform as myNet


def set_seeds(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_inner_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def load_pretrained(model: nn.Module, ckpt_path: str):
    if not ckpt_path:
        raise ValueError("--pretrained_path is required for finetuning.")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"pretrained checkpoint not found: {ckpt_path}")
    utils.load_checkpoint(model, ckpt_path)


def set_trainable_by_policy(model: nn.Module, policy: str):
    m = get_inner_model(model)

    # default: freeze all first
    for p in m.parameters():
        p.requires_grad = False

    if policy == "all":
        for p in m.parameters():
            p.requires_grad = True
        return

    if policy == "kernel_heads_only":
        train_prefixes = [
            "KernelPredictFlow",
            "KernelPredictFlowMask",
            "KernelOutBias",
            "KernelOutWeight",
            "KernelOutkernelx",
            "KernelOutkernely",
            "KernelOutAlpha",
            "KernelOutBeta",
        ]
    elif policy == "kernel_plus_decoder":
        train_prefixes = [
            "Decoder",
            "Convs",
            "feat_extract.3",
            "feat_extract.4",
            "AFFs",
            "FAM1",
            "FAM2",
            "KernelPredictFlow",
            "KernelPredictFlowMask",
            "KernelOutBias",
            "KernelOutWeight",
            "KernelOutkernelx",
            "KernelOutkernely",
            "KernelOutAlpha",
            "KernelOutBeta",
        ]
    else:
        raise ValueError(f"Unsupported freeze policy: {policy}")

    for name, p in m.named_parameters():
        if any(name.startswith(prefix) for prefix in train_prefixes):
            p.requires_grad = True


def count_trainable(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def compute_remaining_schedule(start_epoch: int, num_epochs: int, warmup_epochs: int):
    done_epochs = max(0, start_epoch - 1)
    remain_epochs = max(0, num_epochs - done_epochs)
    # Bound warmup by both original warmup budget and total remaining epochs.
    remain_warmup = min(max(0, warmup_epochs - done_epochs), remain_epochs)
    cosine_phase_epochs = max(0, remain_epochs - remain_warmup)
    # CosineAnnealingLR requires T_max >= 1.
    cosine_tmax = max(1, cosine_phase_epochs)
    return remain_epochs, remain_warmup, cosine_tmax


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    if v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')


def main():
    parser = argparse.ArgumentParser(description="Finetune MISCKernelNet_Deform on custom WindTurbine dataset")

    # data
    parser.add_argument('--train_dir', default='./dataset/WindTurbine', type=str)
    parser.add_argument('--train_meta', default='./dataset/WindTurbine/WindTurbine_train_list.txt', type=str)
    parser.add_argument('--val_dir', default='./dataset/WindTurbine', type=str)
    parser.add_argument('--val_meta', default='./dataset/WindTurbine/WindTurbine_val_list.txt', type=str)

    # io
    parser.add_argument('--model_save_dir', default='./checkpoints_deform', type=str)
    parser.add_argument('--dataset', default='WindTurbine', type=str)
    parser.add_argument('--session', default='MISCFilter_Deform_Wind_FT', type=str)
    parser.add_argument('--pretrained_path', default='./checkpoints_deform/GoPro/MISCFilter_Deform_GoPro/model_best.pth', type=str)

    # training
    parser.add_argument('--patch_size', default=256, type=int)
    parser.add_argument('--num_epochs', default=120, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--val_batch_size', default=16, type=int)
    parser.add_argument('--val_epochs', default=2, type=int)
    parser.add_argument('--print_iters', default=100, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--save_every_epoch', type=str2bool, nargs='?', const=True, default=True)

    # lr
    parser.add_argument('--start_lr', default=5e-5, type=float)
    parser.add_argument('--end_lr', default=1e-6, type=float)
    parser.add_argument('--warmup_epochs', default=3, type=int)

    # deform settings
    parser.add_argument('--use_deform_in_feat', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_deform_in_encoder', type=str2bool, nargs='?', const=True, default=True)

    # finetune policy
    parser.add_argument('--freeze_policy', default='kernel_plus_decoder',
                        choices=['all', 'kernel_heads_only', 'kernel_plus_decoder'])
    parser.add_argument('--resume', action='store_true', help='resume from model_latest.pth in current session dir')

    # env
    parser.add_argument('--gpus', default='0,1', type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.backends.cudnn.benchmark = True
    set_seeds(args.seed)

    model_dir = os.path.join(args.model_save_dir, args.dataset, args.session)
    utils.mkdir(model_dir)
    log_path = os.path.join(model_dir, 'log.txt')

    model_restoration = myNet(
        use_deform_in_feat=args.use_deform_in_feat,
        use_deform_in_encoder=args.use_deform_in_encoder
    )
    model_restoration.cuda()

    load_pretrained(model_restoration, args.pretrained_path)

    if torch.cuda.device_count() > 1:
        model_restoration = nn.DataParallel(model_restoration)

    set_trainable_by_policy(model_restoration, args.freeze_policy)
    total, trainable = count_trainable(get_inner_model(model_restoration))

    print('=' * 70)
    print('Finetune with freeze_policy:', args.freeze_policy)
    print('Total params:', total)
    print('Trainable params:', trainable)
    print('=' * 70)

    optimizer = optim.Adam(trainable_params(model_restoration), lr=args.start_lr, betas=(0.9, 0.999), eps=1e-8)

    start_epoch = 1
    if args.resume:
        resume_ckpt = os.path.join(model_dir, 'model_latest.pth')
        if not os.path.isfile(resume_ckpt):
            raise FileNotFoundError(f'resume requested, but checkpoint not found: {resume_ckpt}')
        utils.load_checkpoint(model_restoration, resume_ckpt)
        start_epoch = utils.load_start_epoch(resume_ckpt) + 1
        utils.load_optim(optimizer, resume_ckpt)

    if start_epoch > args.num_epochs:
        print(f'Current checkpoint epoch already exceeds num_epochs ({start_epoch - 1} > {args.num_epochs}), nothing to train.')
        return

    remain_epochs, remain_warmup, cosine_tmax = compute_remaining_schedule(
        start_epoch=start_epoch,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs
    )

    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_tmax,
        eta_min=args.end_lr
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=remain_warmup,
        after_scheduler=scheduler_cosine
    )

    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss()
    criterion_fft = losses.fftLoss()

    train_dataset = get_training_data(args.train_dir, args.train_meta, {'patch_size': args.patch_size})
    val_dataset = get_validation_data(args.val_dir, args.val_meta, {'patch_size': args.patch_size})

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=True
    )

    with open(log_path, 'a+') as f:
        f.write('=' * 70 + '\n')
        f.write(f'pretrained_path: {args.pretrained_path}\n')
        f.write(f'freeze_policy: {args.freeze_policy}\n')
        f.write(f'total_params: {total}\n')
        f.write(f'trainable_params: {trainable}\n')
        f.write('=' * 70 + '\n')

    best_psnr = 0.0
    best_epoch = 0

    print(f'===> Start Epoch {start_epoch} End Epoch {args.num_epochs}')
    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        model_restoration.train()

        for i, data in enumerate(train_loader):
            for p in model_restoration.parameters():
                p.grad = None

            target_ = data[0].cuda(non_blocking=True)
            input_ = data[1].cuda(non_blocking=True)

            target = kornia.geometry.transform.build_pyramid(target_, 3)
            restored, restored_inter = model_restoration(input_)

            loss_fft = (criterion_fft(restored[0], target[0]) +
                        criterion_fft(restored[1], target[1]) +
                        criterion_fft(restored[2], target[2]))
            loss_char = (criterion_char(restored[0], target[0]) +
                         criterion_char(restored[1], target[1]) +
                         criterion_char(restored[2], target[2]))
            loss_edge = (criterion_edge(restored[0], target[0]) +
                         criterion_edge(restored[1], target[1]) +
                         criterion_edge(restored[2], target[2]))
            loss_char_inter = (criterion_char(restored_inter[0], target[0]) +
                               criterion_char(restored_inter[1], target[1]) +
                               criterion_char(restored_inter[2], target[2]))

            loss = loss_char + loss_char_inter + 0.01 * loss_fft + 0.05 * loss_edge
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % args.print_iters == 0:
                print(f'epoch {epoch} iter {i}')
                print('loss/fft_loss', loss_fft.item())
                print('loss/char_loss', loss_char.item())
                print('loss/edge_loss', loss_edge.item())
                print('loss/iter_loss', loss.item())

        if epoch % args.val_epochs == 0:
            model_restoration.eval()
            psnr_val_rgb = []

            with torch.no_grad():
                for data_val in val_loader:
                    target = data_val[0].cuda(non_blocking=True)
                    input_ = data_val[1].cuda(non_blocking=True)
                    restored, _ = model_restoration(input_)
                    for res, tar in zip(restored[0], target):
                        psnr_val_rgb.append(utils.torchPSNR(res, tar))

            psnr_val = torch.stack(psnr_val_rgb).mean().item() if len(psnr_val_rgb) > 0 else 0.0
            print('val/psnr', psnr_val, epoch)

            if psnr_val > best_psnr:
                best_psnr = psnr_val
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, os.path.join(model_dir, 'model_best.pth'))

            with open(log_path, 'a+') as f:
                f.write(f'[epoch {epoch} PSNR: {psnr_val:.4f} --- best_epoch {best_epoch} Best_PSNR {best_psnr:.4f}]\n')

        scheduler.step()

        lr_now = scheduler.get_lr()[0]
        print('-' * 66)
        print(f'Epoch: {epoch}\tTime: {time.time() - epoch_start:.2f}\tLoss: {epoch_loss:.4f}\tLR {lr_now:.6f}')
        print('-' * 66)

        with open(log_path, 'a+') as f:
            f.write('-' * 66 + '\n')
            f.write(f'Epoch: {epoch}\tTime: {time.time() - epoch_start:.2f}\tLoss: {epoch_loss:.4f}\tLR {lr_now:.6f}\n')
            f.write('-' * 66 + '\n')

        if args.save_every_epoch:
            torch.save({
                'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, os.path.join(model_dir, f'model_epoch_{epoch}.pth'))

        torch.save({
            'epoch': epoch,
            'state_dict': model_restoration.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, os.path.join(model_dir, 'model_latest.pth'))


if __name__ == '__main__':
    main()
