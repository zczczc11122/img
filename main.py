import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
import torch
import logging
import sys
import time
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from config import parser
from dataset import DataSet_Jump
# from model import VitForImageClassification, Resnet
from model import Resnet
from optimizer import WarmupAndOrderedScheduler
from metric import AverageMeter, multi_acc, plot_report, plot_heatmap, write_val_result
from augment import DarkTran
import torch.backends.cudnn as cudnn
import shutil
import matplotlib.pyplot as plt
args = parser.parse_args()
best_acc = 0.0

"""
初始化日志
"""

os.makedirs(os.path.join(args.save_path, args.experiment_pref), exist_ok=True)
logger = logging.getLogger('vit')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s')

handler_file = logging.FileHandler(os.path.join(args.save_path, args.experiment_pref, 'train.log'), mode='w')
handler_file.setFormatter(formatter)
handler_stream = logging.StreamHandler(sys.stdout)
handler_stream.setFormatter(formatter)
logger.addHandler(handler_file)
logger.addHandler(handler_stream)

######################################################################
# Draw Curve
# ------------
x_epoch = []

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []

y_acc = {}
y_acc['train_acc'] = []
y_acc['val_acc'] = []

fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="acc")


draw_flag = True

def draw_curve(current_epoch):
    global draw_flag
    x_epoch.append(current_epoch)

    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')

    ax1.plot(x_epoch, y_acc['train_acc'], 'bo-', label='train')
    ax1.plot(x_epoch, y_acc['val_acc'], 'ro-', label='val')

    if draw_flag == True:
        ax0.legend()
        ax1.legend()
        draw_flag = False
    fig.savefig(os.path.join(".", 'train.jpg'))


def _init_model():
    global best_acc
    model = Resnet(args)
    if args.use_gpu:
        logger.info("Let's use {} GPUs !".format(args.num_gpus))
        model = torch.nn.DataParallel(model).cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    return model



def main():
    global best_acc
    cudnn.benchmark = True
    model = _init_model()

    cover_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
    data_transform = {
        "train": transforms.Compose([transforms.RandomSizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     cover_normalize]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   cover_normalize])}

    train_dataset = DataSet_Jump(args.file_path, "train",
                                 args.pre_path,
                                 data_transform["train"])
    val_dataset = DataSet_Jump(args.file_path, "val",
                               args.pre_path,
                               data_transform["val"])
    logger.info(f"train_dataset{len(train_dataset)}")
    logger.info(f"val_dataset{len(val_dataset)}")
    if not args.use_gpu:
        args.num_gpus = 1

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size * args.num_gpus,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=args.workers,
                              drop_last=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size_val * args.num_gpus,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.workers,
                            drop_last=False)

    model_params = [p for p in model.module.parameters()]
    optimizer = torch.optim.AdamW(params=model_params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    num_steps_per_epoch = len(train_dataset) // (args.batch_size * args.num_gpus)
    model_scheduler = WarmupAndOrderedScheduler(optimizer=optimizer,
                                              num_steps_per_epoch=num_steps_per_epoch,
                                              num_warmup_epochs=args.num_warmup_epochs,
                                              epoch_start=args.start_epoch)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_acc = \
            train(train_loader, model, criterion, optimizer, model_scheduler, epoch)

        y_loss['train'].append(train_loss)
        y_acc['train_acc'].append(train_acc)

        val_loss, val_acc = validate(val_loader, model, criterion, epoch, "val")

        y_loss['val'].append(val_loss)
        y_acc['val_acc'].append(val_acc)

        draw_curve(epoch)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        _save_checkpoint({'epoch': epoch + 1, 'arch': args.arch,
                          'state_dict': model.state_dict(), 'best_acc': best_acc},
                         is_best,
                         '{}_checkpoint.pth.tar'.format('last'))



def train(train_loader, model, criterions, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    model.train()

    end = time.time()
    for step, (feature, label, img_path) in enumerate(train_loader):
        data_time.update(time.time() - end)

        feature = feature.cuda()
        label = label.cuda()
        outputs = model(feature)
        loss = criterions(outputs, label)

        batch_size = feature.size(0)
        result = multi_acc(outputs.data, label)

        accs.update(result.item(), batch_size)
        losses.update(loss.data.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            logger.info('Epoch: [{epoch}][{step}/{len}]\t'
                        'lr: {lr:.8f}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch=epoch, step=step,
                                                                   len=len(train_loader),
                                                                   lr=optimizer.param_groups[-1]['lr'],
                                                                   batch_time=batch_time,
                                                                   data_time=data_time,
                                                                   loss=losses, acc=accs))
    return losses.avg, accs.avg

def validate(val_loader, model, criterion, epoch, mode):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    test_list = []
    pred_list = []
    pred_list_value = []
    path_list = []

    model.eval()
    end = time.time()
    with torch.no_grad():
        for step, (feature, label, img_path) in enumerate(val_loader):
            path_list.extend(list(img_path))
            feature = feature.cuda()
            test_list.extend(label.cpu().numpy().tolist())

            label = label.cuda()
            outputs = model(feature)

            _, pred_tags = torch.max(outputs, dim=1)
            pred_value = torch.softmax(outputs, dim=1)
            pred_value, _  = torch.max(pred_value, dim=1)
            pred_list.extend(pred_tags.cpu().numpy().tolist())
            pred_list_value.extend(pred_value.cpu().numpy().tolist())

            loss = criterion(outputs, label)
            result = multi_acc(outputs.data, label)

            batch_size = feature.size(0)
            losses.update(loss.data.item(), batch_size)
            accs.update(result.item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()
        logger.info('{mode}: [{epoch}/{total_epoch}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(mode=mode,
                                                               epoch=epoch,
                                                               total_epoch=args.epochs,
                                                               batch_time=batch_time,
                                                               loss=losses,
                                                               acc=accs))
        logger.info(f'{mode} Results: Acc {accs.avg:.3f} Loss {losses.avg:.5f}')

    plot_heatmap({'y_test': test_list, 'y_pred': pred_list},
                 save_path=os.path.join(args.save_path, args.experiment_pref,
                                        f'heatmap_{epoch}_{accs.avg}_{mode}.png'))
    plot_report({'y_test': test_list, 'y_pred': pred_list},
                save_path=os.path.join(args.save_path, args.experiment_pref,
                                       f'report_{epoch}_{accs.avg}_{mode}.json'))
    write_val_result({'path': path_list, 'y_test': test_list, 'y_pred': pred_list, 'y_value': pred_list_value},
                     save_path=os.path.join(args.save_path, args.experiment_pref,
                                            f'case_result_{epoch}_{accs.avg}_{mode}.csv'))


    return losses.avg, accs.avg


def _save_checkpoint(state, is_best, checkpoint_filename):
    target_path = os.path.join(args.save_path, args.experiment_pref)
    os.makedirs(target_path, exist_ok=True)
    torch.save(state, os.path.join(target_path, checkpoint_filename))
    if is_best:
        best_name = 'best.pth.tar'
        shutil.copyfile(os.path.join(target_path, checkpoint_filename),
                        os.path.join(target_path, best_name))

if __name__ == '__main__':
    main()

