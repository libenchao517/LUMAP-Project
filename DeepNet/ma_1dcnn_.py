################################################################################
# 本文件用于实现MA 1DCNN网络
################################################################################
# 导入模块
from torch.utils import data as da
from timm.loss import LabelSmoothingCrossEntropy
import argparse
from .ma1dcnn import MA1DCNN
import torch
import numpy as np
################################################################################
# 定义网络
class MA_1DCNN:
    """
    Wang H, Liu Z, Peng D, et al.
    Understanding and learning discriminant features based on multiattention 1DCNN for wheelset bearing fault diagnosis[J].
    IEEE Transactions on Industrial Informatics, 2019, 16(9): 5735-5745.
    """
    def __init__(
            self,
            sample_height=1,
            sample_weight=10,
            num_classes=10,
            epoch=500,
            device=None
    ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.epoch = epoch
        self.sample_height = sample_height
        self.sample_weight = sample_weight
        self.num_classes = num_classes

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Train')
        parser.add_argument('--data_dir', type=str, default= "data\\5HP", help='the directory of the data')
        parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
        parser.add_argument('--batch_size', type=int, default=128, help='batchsize of the training process')
        parser.add_argument('--step_len', type=list, default=range(210, 430, 10), help='the weight decay')
        parser.add_argument('--sample_len', type=int, default=420, help='the learning rate schedule')
        parser.add_argument('--rate', type=list, default=[0.7, 0.15, 0.15], help='')
        parser.add_argument('--acces', type=list, default=[], help='initialization list')
        parser.add_argument('--epochs', type=int, default=self.epoch, help='max number of epoch')
        parser.add_argument('--losses', type=list, default=[], help='initialization list')
        args = parser.parse_args()
        return args

    def fit(
            self,
            X_train,
            X_test,
            T_train,
            T_test
    ):
        args = self.parse_args()
        Train = Dataset(X_train, T_train)
        Test = Dataset(X_test, T_test)
        train_loader = da.DataLoader(Train, batch_size=args.batch_size, shuffle=True)
        test_loader = da.DataLoader(Test, batch_size=args.batch_size, shuffle=False)

        # 加载模型
        model = MA1DCNN(self.num_classes, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = LabelSmoothingCrossEntropy()
        model.train()
        for epoch in range(args.epochs):
            for step, (img, label) in enumerate(train_loader):
                img = img.float().to(self.device)
                label = label.long().to(self.device)
                out = model(img)
                out = torch.squeeze(out).float()
                loss = criterion(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 37 == 0:
                    print(epoch, step, "loss: ", float(loss))
        model.eval()

        all_preds = []
        for img, label in test_loader:
            img = img.float().to(self.device)
            label = label.long().to(self.device)
            out = model(img)
            out = torch.squeeze(out).float()
            _, pred = out.max(1)
            all_preds.extend(pred.tolist())
        return all_preds

    def fit_transform(
            self,
            X_train,
            X_test,
            T_train,
            T_test
    ):
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        T_train = T_train.astype(int)
        T_test = T_test.astype(int)
        T_pred = self.fit(X_train, X_test, T_train, T_test)
        return T_pred

################################################################################
# 定义数据集
class Dataset(da.Dataset):
    def __init__(self, X, y):
        self.Data = X
        self.Label = y
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label
    def __len__(self):
        return len(self.Data)
