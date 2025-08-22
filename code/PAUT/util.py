import shutil

import matplotlib.pyplot as plt

from code.PAUT import opt
import os


# 创建基本文件夹
def mk_basic_dir(project_name):
    train_res_dir = f"{opt.base_res_dir}/{project_name}"
    if not os.path.exists(train_res_dir):
        os.mkdir(train_res_dir)
    else:
        for i in range(100):
            new_train_res_dir = train_res_dir + str(i)
            if not os.path.exists(new_train_res_dir):
                os.mkdir(new_train_res_dir)
                train_res_dir = new_train_res_dir
                break
    # 创建存放model文件夹
    model_dir = f'{train_res_dir}/model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    return train_res_dir


# 画出并保存train val结果
def draw_process(history, save_path):
    print('绘制曲线中...')
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.subplot(2, 1, 2)
    plt.plot(history['train_accuracy'], label='train_acc')
    plt.plot(history['val_accuracy'], label='val acc')
    plt.legend()
    plt.title('Model Accuracy')
    plt.savefig(save_path)
    print('曲线绘制结束')


# 保存loss最小的model
def find_best_model(model_dir):
    print('find best model...')
    models = os.listdir(model_dir)
    assert models != []
    min_loss = 100
    best_model = None
    for model in models:
        loss = float(model.split('-')[-1].replace('.pth', ''))
        if loss < min_loss:
            min_loss = loss
            best_model = model
    if best_model:
        shutil.copyfile(os.path.join(model_dir, best_model), os.path.join(model_dir, 'best.pth'))
        print('save best model!')


