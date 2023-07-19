import argparse

parser = argparse.ArgumentParser(description="vit train")

# ========================= Base Configs =========================
parser.add_argument('--file-path', type=str, help='datatset info file path',
                    default='../炭火图片标注_4.xlsx')
parser.add_argument('--pre-path', type=str, help='datatset info file path',
                    default='../fire_img')
parser.add_argument('--file-path-v2', type=str, help='datatset info file path',
                    default='../跳楼数据标注_3期.xlsx')
parser.add_argument('--pre-path-v2', type=str, help='datatset info file path',
                    default='../jump_img_v2')



parser.add_argument('--num-class', type=int, help='0 jump 1 no_jump', default=2)
parser.add_argument('--arch', type=str, help='datatset info file path',
                    default="resnet50")

# ========================= Learning Configs =========================
parser.add_argument('--batch-size', default=32, type=int, help='mini-batch size')
parser.add_argument('--batch-size-val', default=64, type=int, help='mini-batch size')
parser.add_argument('--epochs', default=14, type=int, metavar='N',
                    help='number of total epochs to run')

# parser.add_argument('--dropout', '--do', default=0.5, type=float,
#                     metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', default=0.0000, type=float, help='weight decay')
# parser.add_argument('--divide-every-n-epochs', default=1, type=int, help='learning rate decay every n epochs')
# parser.add_argument('--lr-decay-rate', default=0.9, type=float, help='learning rate decay rate')
parser.add_argument('--num-warmup-epochs', default=1, type=float, help='learning rate decay rate')

# ========================= Monitor Configs =========================
parser.add_argument('--print-freq', '-p', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')

# ========================= Runtime Configs =========================
parser.add_argument('--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--num-gpus', type=int, default=7)
parser.add_argument('--use-gpu', default=True, action='store_false')

# ========================= Checkpoints Configs =========================
parser.add_argument('--resume', default='',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=False, type=bool, help='evaluate model on validation set')
parser.add_argument('--save_path', type=str, default=".t/checkpoints_top_v1")
parser.add_argument('--experiment_pref', type=str, default="20220316_train")
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')


