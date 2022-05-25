import torch
import os
import torch.nn as nn

from ViT import ViT
from ImageEmbed import ImageEmbed
from SA1 import SAHead1
from SA2 import SAHead2
from SA3 import SAHead3
from SA4 import SAHead4
from SA5 import SAHead5
from SelfAttention import SelfAttention
from ConvBlock import ConvBlock
from Classify import Classify
from ViTEncoder import ViTEncoder
from dataloader import whale_dolphin
from torch.utils.data import DataLoader


def train():
    # 1. load dataset
    root = 'dataset/train'
    batch_size = 16
    train_data = whale_dolphin(root, train=True)
    val_data = whale_dolphin(root, train=False)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # 2.load model
    #     num_classes = 2
    #     img_channels = 3
    #     img_size = 128
    #     heads_num = 8
    #     patch_size = 32
    model = ViT(ImageEmbed, SAHead1, SAHead2, SAHead3, SAHead4, SAHead5,
                SelfAttention, ConvBlock, Classify, ViTEncoder,
                8, 3, 40, 2, 11)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to('cuda')

    # 3.prepare hyperparameters
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 20

    # 4.train
    val_acc_list = []
    out_dir = "results/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            length = len(train_dataloader)
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(images)  # torch.size([batch_size, num_class])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1), 100. * correct / total))

        print("Waiting Val...")
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            for batch_idx, (images, labels) in enumerate(val_dataloader):
                model.eval()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            print('Val\'s ac is: %.3f%%' % (100 * correct / total))

            acc_val = 100. * correct / total
            val_acc_list.append(acc_val)

        torch.save(model.state_dict(), out_dir + "last.pt")
        if acc_val == max(val_acc_list):
            torch.save(model.state_dict(), out_dir + "best.pt")
            print(f"save epoch {epoch} model")


train()