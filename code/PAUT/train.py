import time
import argparse
import copy

from code.PAUT.ImgDataset import *
from util import *
from code.PAUT import opt

parser = argparse.ArgumentParser(description="train")
parser.add_argument("--name", type=str, help="name", required=True)
parser.add_argument("--epoch", type=int, help="epoch", default=60)
parser.add_argument("--batch_size", type=int, help="batch_size", default=32)
parser.add_argument("--datasets", type=str, help="数据集路径", default='')
parser.add_argument("--lr", type=float, help="学习率", default=0.00005)
parser.add_argument("--net", type=str, help="根据dict选择网络模型", default='net')
parser.add_argument("--pre", action='store_true', help="是否为预训练模型")
parser.add_argument("--pre_path", type=str, help="预训练模型路径")
parser.add_argument("--freeze", action='store_true', help="是否冻结前面几层")
parser.add_argument("--val", action='store_true', help="是否需要验证集")
parser.add_argument("--former", action='store_true', help="是否需要前置切片")
parser.add_argument("--extra", action='store_true', help="是否有额外特征")
parser.add_argument("--att", type=str, help="注意力机制名字")
parser.add_argument("--dr", type=float, help="drop_rate", default=0.)

args = parser.parse_args()
project_name = args.name
num_epoch = args.epoch
batch_size = args.batch_size
learning_rate = args.lr
NET_KEY = args.net
PRETRAINED = args.pre
PRETRAINED_PATH = args.pre_path
FREEZE = args.freeze
VAL = args.val
FORMER = args.former
ATT = args.att
DROP_RATE = args.dr
EXTRA = args.extra
CUDA_INDEX = opt.CUDA_INDEX

train_res_dir = mk_basic_dir(project_name)

if __name__ == '__main__':
    datasets_path = f"{opt.workspace_dir}{args.datasets}" if args.datasets != '' else opt.workspace_dir
    # 分别将 training set、validation set、testing set 用 readfile 函式读进来
    print("Reading data")
    print("...")
    print(f"从{datasets_path}加载数据集...")
    train_x, train_y, train_extra = readfile(os.path.join(datasets_path, "training"), True, extra=EXTRA)
    print("Size of training data = {}".format(len(train_x)))
    val_x, val_y, val_extra = readfile(os.path.join(datasets_path, "validation"), True, extra=EXTRA)
    print("Size of validation data = {}".format(len(val_x)))
    test_x, test_y, test_extra = readfile(os.path.join(datasets_path, "testing"), True, extra=EXTRA)
    print("Size of testing data = {}".format(len(test_x)))
    print("Reading data complicated")

    ''' Dataset '''
    print("Dataset")
    print("...")
    # training 时做 data augmentation
    # transforms.Compose 将图像操作串联起来
    train_transform = transforms.Compose([
        transforms.ToPILImage(),

        transforms.RandomHorizontalFlip(),  # 随机将图片水平翻转
        transforms.RandomRotation(5),  # 随机旋转图片 (-15,15)

        # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
        # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # 高斯模糊
        # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1), ratio=(0.75, 1.3333333333333333)),


        transforms.ToTensor(),  # 将图片转成 Tensor, 并把数值normalize到[0,1](data normalization)

        # 随机擦除部分区域
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # testing 时不需做 data augmentation
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if VAL:
        train_set = ImgDataset(train_x, train_y, train_transform, extra=train_extra)
        val_set = ImgDataset(val_x, val_y, test_transform, extra=val_extra)
    else:
        # 结合train和val
        train_x = np.concatenate((train_x, val_x), axis=0)
        train_y = np.concatenate((train_y, val_y), axis=0)
        if EXTRA:
            train_extra = np.concatenate((train_extra, val_extra), axis=0)
        train_set = ImgDataset(train_x, train_y, train_transform, extra=train_extra)
        val_set = ImgDataset(test_x, test_y, test_transform, extra=test_extra)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    print("Dataset complicated")

    ''' Training '''
    print("Training")
    print("...")
    # 获得model
    model = opt.get_model_by_str(NET_KEY, pretrained=PRETRAINED, pretrain_path=PRETRAINED_PATH, freeze=FREEZE, att=ATT, drop_rate=DROP_RATE)
    model.cuda(CUDA_INDEX)

    loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optimizer 使用 Adam
    # 定义 AdamW 优化器
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    # 定义 AdamW 优化器
    learning_rate = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # 定义 CosineAnnealingLR 学习率调度器
    # T_max = 100  # 学习率衰减的最大周期数
    # scheduler = CosineAnnealingLR(optimizer, T_max=T_max)

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }
    best_acc = None
    best_model = None
    start_time = time.time()

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
            train_pred = model(data[0].cuda(CUDA_INDEX)) if not EXTRA \
                else model(data[0].cuda(CUDA_INDEX), data[2].cuda(CUDA_INDEX))
            batch_loss = loss(train_pred, data[1].cuda(CUDA_INDEX))  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
            batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
            optimizer.step()  # 以 optimizer 用 gradient 更新參數值

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

        # 更新学习率
        scheduler.step()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].cuda(CUDA_INDEX)) if not EXTRA \
                    else model(data[0].cuda(CUDA_INDEX), data[2].cuda(CUDA_INDEX))
                batch_loss = loss(val_pred, data[1].cuda(CUDA_INDEX))

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()

            res_val_acc = val_acc / val_set.__len__()
            res_val_loss = val_loss / val_set.__len__()
            # 將結果 print 出來
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                  (epoch + 1, num_epoch, time.time() - epoch_start_time, train_acc / train_set.__len__(),
                   train_loss / train_set.__len__(), res_val_acc, res_val_loss))
            if not VAL:
                if best_acc is None or res_val_acc > best_acc:
                    best_acc = res_val_acc
                    best_model = copy.deepcopy(model.state_dict())
                    # 保存每一轮的模型
                    model_name = '{}-{}-{}-{}.pth'.format(
                        NET_KEY,
                        epoch + 1,
                        round(res_val_acc, 4),
                        round(res_val_loss, 6))
                    torch.save(model.state_dict(), f"{train_res_dir}/model/{model_name}")
                if epoch == num_epoch - 1:
                    torch.save(model.state_dict(), f"{train_res_dir}/model/last.pth")
            # 保存到历史，用于绘制曲线
            history['epoch'].append(epoch + 1)
            history['train_accuracy'].append(train_acc / train_set.__len__())
            history['train_loss'].append(train_loss / train_set.__len__())
            history['val_accuracy'].append(val_acc / val_set.__len__())
            history['val_loss'].append(val_loss / val_set.__len__())
    # 绘制曲线并保存
    draw_process(history=history, save_path=f'{train_res_dir}/process.jpg')

    if not VAL:
        # 找到并保存最好的model
        torch.save(best_model, f"{train_res_dir}/model/best.pth")

    print("Training complicated")
    print('use time %2.2f h' % ((time.time() - start_time) / 60 / 60))
