"""
Visualized the embedding feature of the pre-train model on the training set.
"""
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

import sys
sys.path.append('/home/tao/code/cls/pytorch-image-models')
from timm.data import create_dataset, create_loader
from timm.models import create_model, load_checkpoint

import numpy as np
import argparse
import json


# 使用钩子函数来获取指定层的输出
outputs = {}

def hook_fn(module, input, output):
    outputs["global_pool"] = output

def emb_fea(model, dataloader, args):
    # model to evaluate mode
    model.global_pool.register_forward_hook(hook_fn)
    model.eval()
    EMB = {}

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()

            # compute output
            # emb_fea, logits = model(images, embed=True)
            logits = model(images)
            emb_fea = outputs["global_pool"]

            for emb, i in zip(emb_fea, labels):
                i = i.item()
                assert len(emb) == args.emb_size
                if str(i) in EMB:
                    for j in range(len(emb)):
                        EMB[str(i)][j].append(round(emb[j].item(), 4))
                else:
                    EMB[str(i)] = [[] for _ in range(len(emb))]
                    for j in range(len(emb)):
                        EMB[str(i)][j].append(round(emb[j].item(), 4))

    
    for key, value in EMB.items():
        for i in range(args.emb_size):
            EMB[key][i] = round(np.array(EMB[key][i]).mean(), 4)
    
    return EMB


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualized the embedding feature of the model on the train set.')
    parser.add_argument("--emb_size", type=int, default=2048, help="emb fea size")
    parser.add_argument("--dataset", type=str, default='cifar10')
    parser.add_argument("--batch_size", type=int, default=64, help="total batch size")
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--teacher-num-classes', type=int, default=10, metavar='N',
                   help='number of label classes (Model default if None)')
    parser.add_argument('--teachermodel', default='resnet101', type=str, metavar='TEACHERMODEL',
                   help='Name of model to train (default: "resnet50")')
    parser.add_argument('--teacher-initial-checkpoint', default='results/resnet101/20241103-160549-resnet101-224/model_best.pth.tar', type=str, metavar='PATH',
                   help='Load this checkpoint into model after initialization (default: none)')
    parser.add_argument('--data-dir', default='/home/tao/dataset/cifar10/',metavar='DIR',
                    help='path to dataset (root dir)')
    parser.add_argument('--train-split', metavar='NAME', default='train/',
                   help='dataset train split (default: train)')

    args = parser.parse_args()

    
    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=False,
    )
    loader_train = create_loader(
        dataset_train,
        input_size=(3,224,224),
        batch_size=64,
        is_training=False,
        re_prob=0.,
        re_mode='pixel',
        re_count=1,
        re_split=False,
        scale=[0.08, 1.0],
        ratio=[3. / 4., 4. / 3.],
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        num_workers=4,
        distributed=False,
        worker_seeding='all',
    )
    teacher_model = create_model(
        args.teachermodel,
        pretrained=False,
        num_classes=args.teacher_num_classes,
        in_chans=3,
        scriptable='torchscript',
    )
    load_checkpoint(teacher_model, args.teacher_initial_checkpoint)


    teacher_model = teacher_model.cuda()

    emb = emb_fea(model=teacher_model, dataloader=loader_train, args=args)
    emb_json = json.dumps(emb, indent=4)
    with open("./tools/run/{}_embedding_fea_{}.json".format(args.dataset, args.teachermodel), 'w', encoding='utf-8') as f:
        f.write(emb_json)
    f.close()