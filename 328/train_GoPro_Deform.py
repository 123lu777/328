import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data.data_RGB import get_training_data, get_validation_data

# =============================================
# 导入可变形卷积版本的模型
# =============================================
from models.MISCFilterNet_Deform import MISCKernelNet_Deform as myNet

from loss import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tools.get_parameter_number import get_parameter_number
import kornia

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
start_epoch = 1


# =============================================
# 所有参数在这里直接配置，右键即可运行
# =============================================
class Config:
    # 数据集路径
    train_dir = './dataset/GOPRO_Large'
    train_meta = './dataset/GOPRO_Large/GOPRO_train_list.txt'
    val_dir = './dataset/GOPRO_Large'
    val_meta = './dataset/GOPRO_Large/GOPRO_test_list.txt'

    # 模型保存路径
    model_save_dir = './checkpoints_deform'
    dataset = 'GoPro'
    session = 'MISCFilter_Deform_GoPro'

    # 训练参数
    patch_size = 256  # [GoPro, HIDE, RealBlur]=256, [DPDD]=512
    num_epochs = 6000
    batch_size = 8  # 可变形卷积显存占用更大，建议从8开始
    val_epochs = 10
    print_epochs = 2

    # 可变形卷积设置
    use_deform_in_feat = True  # 是否在特征提取中使用可变形卷积
    use_deform_in_encoder = True  # 是否在编码器/解码器中使用可变形卷积

    # 恢复训练设置
    RESUME = True  # 首次训练设为 False，恢复训练设为 True
    Pretrain = False
    model_pre_dir = ''


args = Config()

dataset = args.dataset
session = args.session
patch_size = args.patch_size

model_dir = os.path.join(args.model_save_dir, dataset, session)
utils.mkdir(model_dir)
log_dir = os.path.join(args.model_save_dir, dataset, session, 'log.txt')

train_dir = args.train_dir
val_dir = args.val_dir

train_meta = args.train_meta
val_meta = args.val_meta

num_epochs = args.num_epochs
batch_size = args.batch_size
val_epochs = args.val_epochs

start_lr = 2e-4
end_lr = 1e-6

######### Model ###########
model_restoration = myNet(
    use_deform_in_feat=args.use_deform_in_feat,
    use_deform_in_encoder=args.use_deform_in_encoder
)

# print number of model
total_num, trainable_num = get_parameter_number(model_restoration)
print('=' * 60)
print('Model: MISCKernelNet with Deformable Convolution')
print('Use Deform in Feature Extraction:', args.use_deform_in_feat)
print('Use Deform in Encoder/Decoder:', args.use_deform_in_encoder)
print('=' * 60)
print('Total:  ', total_num)
print('Trainable: ', trainable_num)
with open(log_dir, "a+") as f:
    f.write('=' * 60 + '\n')
    f.write('Model: MISCKernelNet with Deformable Convolution\n')
    f.write('Use Deform in Feature Extraction: {}\n'.format(args.use_deform_in_feat))
    f.write('Use Deform in Encoder/Decoder: {}\n'.format(args.use_deform_in_encoder))
    f.write('=' * 60 + '\n')
    f.write('Total: {}\n'.format(total_num))
    f.write('Trainable: {}\n'.format(trainable_num))

model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

optimizer = optim.Adam(model_restoration.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs - warmup_epochs, eta_min=end_lr)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

RESUME = args.RESUME
Pretrain = args.Pretrain
model_pre_dir = args.model_pre_dir

######### Pretrain ###########
if Pretrain:
    utils.load_checkpoint(model_restoration, model_pre_dir)
    print('------------------------------------------------------------------------------')
    print("==> Retrain Training with:  " + model_pre_dir)
    print('------------------------------------------------------------------------------')

######### Resume ###########
if RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

######### Loss ###########
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()
criterion_fft = losses.fftLoss()

######### DataLoaders ###########
train_dataset = get_training_data(train_dir, train_meta, {'patch_size': patch_size})
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False,
                          pin_memory=True)

val_dataset = get_validation_data(val_dir, val_meta, {'patch_size': patch_size})
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=False,
                        pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, num_epochs + 1))
print('===> Loading datasets')
with open(log_dir, "a+") as f:
    f.write('===> Start Epoch {} End Epoch {} \n'.format(start_epoch, num_epochs + 1))
    f.write('===> Loading datasets\n')

best_psnr = 0
best_epoch = 0
iter = 0

for epoch in range(start_epoch, num_epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for i, data in enumerate(train_loader, 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target_ = data[0].cuda()
        input_ = data[1].cuda()
        target = kornia.geometry.transform.build_pyramid(target_, 3)
        restored, restored_inter = model_restoration(input_)

        loss_fft = criterion_fft(restored[0], target[0]) + criterion_fft(restored[1], target[1]) + criterion_fft(
            restored[2], target[2])
        loss_char = criterion_char(restored[0], target[0]) + criterion_char(restored[1], target[1]) + criterion_char(
            restored[2], target[2])
        loss_edge = criterion_edge(restored[0], target[0]) + criterion_edge(restored[1], target[1]) + criterion_edge(
            restored[2], target[2])
        loss_char_inter = criterion_char(restored_inter[0], target[0]) + criterion_char(restored_inter[1],
                                                                                        target[1]) + criterion_char(
            restored_inter[2], target[2])

        loss = loss_char + loss_char_inter + 0.01 * loss_fft + 0.05 * loss_edge
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        iter += 1

        # 每100个iteration打印一次
        if i % 100 == 0:
            print('epoch', epoch, 'iter', i)
            print('loss/fft_loss', loss_fft.item())
            print('loss/char_loss', loss_char.item())
            print('loss/edge_loss', loss_edge.item())
            print('loss/iter_loss', loss.item())

    if epoch % args.print_epochs == 0:
        print('loss/epoch_loss', epoch_loss, epoch)

    #### Evaluation ####
    if epoch % val_epochs == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()

            with torch.no_grad():
                restored, _ = model_restoration(input_)

            for res, tar in zip(restored[0], target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        print('val/psnr', psnr_val_rgb, epoch)
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_best.pth"))

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))
        with open(log_dir, "a+") as f:
            f.write("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f] \n" % (
                epoch, psnr_val_rgb, best_epoch, best_psnr))

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(log_dir, "a+") as f:
        f.write("------------------------------------------------------------------\n")
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f} \n".format(epoch,
                                                                                       time.time() - epoch_start_time,
                                                                                       epoch_loss,
                                                                                       scheduler.get_lr()[0]))
        f.write("------------------------------------------------------------------\n")

    # =============================================
    # 每轮都保存权重
    # =============================================
    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

    # 保存最新的权重（用于恢复训练）
    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))