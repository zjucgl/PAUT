import argparse

import matplotlib.pyplot as plt

from code.PAUT.ImgDataset import *
from code.PAUT import opt
from torch.nn import functional as F
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--name", type=str, help="name", required=True)
parser.add_argument("--batch_size", type=int, help="batch_size", default=32)
parser.add_argument("--datasets", type=str, help="数据集路径", default='')
parser.add_argument("--model_path", type=str, help="model_path")
parser.add_argument("--net", type=str, help="根据dict选择网络模型", default='net')
parser.add_argument("--pre", action='store_true', help="是否为预训练模型")
parser.add_argument("--former", action='store_true', help="是否需要前置切片")
parser.add_argument("--att", type=str, help="注意力机制名字")

args = parser.parse_args()
project_name = args.name
# project_name = 'res50'
batch_size = args.batch_size
model_path = args.model_path
NET_KEY = args.net
# NET_KEY = 'resnet50'
PRETRAINED = args.pre
FORMER = args.former
ATT = args.att

CUDA_INDEX = opt.CUDA_INDEX

train_res_dir = f"{opt.base_res_dir}/{project_name}"
if not model_path:
    model_path = f"{train_res_dir}/model/best.pth"
    print('-----')
    print(model_path)

if __name__ == '__main__':
    datasets_path = f"{opt.workspace_dir}{args.datasets}" if args.datasets != '' else opt.workspace_dir
    print(f'从{datasets_path}加载数据集...')
    ''' Testing '''
    print("Testing")
    print("...")

    test_x, test_y, test_former = readfile(os.path.join(datasets_path, "testing"), True, former=FORMER)
    sides = []

    print("Size of test data = {}".format(len(test_x)))
    # testing 时不需做 data augmentation
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_set = ImgDataset(test_x, test_y, transform=test_transform, former_slice=test_former)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # 加载模型
    model_best = opt.get_model_by_str(NET_KEY, pretrained=PRETRAINED, att=ATT)
    model_best.cuda(CUDA_INDEX)
    # 加载权重
    print("从 {} 加载权重".format(model_path))
    model_best.load_state_dict(torch.load(model_path))

    model_best.eval()
    prediction = []

    pred_list = []
    label_list = []
    features = []


    def hook_fn(module, input, output):
        features.append(output)


    # 在 `avgpool` 层上注册 hook
    hook = model_best.avgpool.register_forward_hook(hook_fn)

    correct = 0
    total = 0
    class_correct = list(0. for i in range(len(opt.cls)))
    class_total = list(0. for i in range(len(opt.cls)))

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0], data[1]
            images, labels = images.cuda(CUDA_INDEX), labels.cuda(CUDA_INDEX)
            outputs = model_best(images) if not FORMER \
                else model_best(data[0].cuda(CUDA_INDEX), data[2].cuda(CUDA_INDEX))

            # 将概率转为[0, 100]的整数
            outputs = F.softmax(outputs, dim=1)
            outputs = torch.round(outputs * 100)

            pred_list.extend(outputs.cpu().numpy())
            label_list.extend(labels.cpu().numpy())

    label_list = np.array(label_list)
    predicted = np.argmax(pred_list, axis=1)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(label_list, predicted)
    # 按行归一化（每行和为1，显示的是每个真实类别的预测分布）
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    # label_txt = os.listdir(os.path.join(f"/home/admin/datasets/PAUT/datasets-s-{args.datasets}", 'testing'))
    label_txt = ['ND', 'PD', 'TD']
    sns.heatmap(conf_matrix_percent, annot=True, fmt=".2%", cmap="Blues", xticklabels=label_txt, yticklabels=label_txt)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.savefig(f'{train_res_dir}/matrix-{train_res_dir.split("/")[-1]}.png')
    # plt.show()

    # 取消 hook
    hook.remove()
    labels = label_list
    # **合并所有 batch 的 features**
    features = torch.cat(features, dim=0)  # (total_samples, 512)
    features = features.view(features.size(0), -1).cpu().numpy()
    # --- 新增：先用 PCA 将高维特征降到50维 ---
    features = PCA(n_components=min(50, features.shape[1]), random_state=42).fit_transform(features)

    # 设定 t-SNE 参数
    param_grid = [
        dict(perplexity=30, learning_rate=200, early_exaggeration=12),
        dict(perplexity=30, learning_rate=500, early_exaggeration=12),
        dict(perplexity=50, learning_rate=500, early_exaggeration=12),
        dict(perplexity=10, learning_rate=200, early_exaggeration=24),
    ]

    for index, p in enumerate(param_grid):
        tsne = TSNE(
            n_components=2,
            perplexity=p["perplexity"],
            learning_rate=p["learning_rate"],
            early_exaggeration=p["early_exaggeration"],
            n_iter=2000,
            init="pca",
            metric="cosine",
            random_state=42
        )

        embedded_features = tsne.fit_transform(features)
        plt.figure(figsize=(8, 6))

        label_names = np.array([label_txt[int(l)] for l in labels])
        ax = sns.scatterplot(
            x=embedded_features[:, 0],
            y=embedded_features[:, 1],
            hue=label_names,
            palette="bright",
            alpha=0.7
        )
        ax.legend(title="Class", bbox_to_anchor=(1, 1))

        plt.savefig(f'{train_res_dir}/t-SNE-{train_res_dir.split("/")[-1]}-{index+1}.png')

    # 计算分类报告
    report = classification_report(label_list, predicted, digits=4, target_names=label_txt)
    print('-' * 40)
    print(report)

    conf_matrix_array = np.array(conf_matrix)
    print(conf_matrix_array)
    # 将测试结果写入txt
    with open(f'{train_res_dir}/test_res.txt', 'w') as f:
        f.write(report)

    print("Testing complicated")
