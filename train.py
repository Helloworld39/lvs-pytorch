import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import data
import shutil
from model import UNet2D, UNet3D


class Train:
    def __init__(self, **kwargs):
        # device设定
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            if torch.cuda.is_available():
                self.device = torch.device(0)
            else:
                self.device = torch.device('cpu')

        # model设定
        if 'model' in kwargs:
            if kwargs['model'] == 'unet':
                self.model = UNet2D().to(self.device)
            elif kwargs['model'] == 'unet3d':
                self.model = UNet3D().to(self.device)
            else:
                print('错误：不存在的模型。')
                exit(3)
        else:
            print('错误：Train中缺少必要参数[model]，例如Train(model="unet", ...)')
            exit(2)

        # criterion设定
        if 'criterion' in kwargs:
            if kwargs['criterion'] == 'bce':
                self.criterion = nn.BCELoss().to(self.device)
            elif kwargs['criterion'] == 'ce':
                self.criterion = nn.CrossEntropyLoss().to(self.device)
            else:
                print('错误：不存在的损失函数。')
                exit(3)
        else:
            print('错误：Train中缺少必要参数[criterion]，例如Train(criterion="bce", ...)')
            exit(2)

        # optimizer设定
        if 'optimizer' in kwargs:
            # 设定学习率，默认为1e-3
            self.lr = kwargs['lr'] if 'lr' in kwargs else 1e-3
            if kwargs['optimizer'] == 'adam':
                self.optimizer = optim.Adam(self.model.parameters(), self.lr)
            else:
                print('错误：不存在的优化器。')
                exit(3)
        else:
            print('错误：Train中缺少必要参数[optimizer]，例如Train(optimizer="adam", ...)')
            exit(2)

        # scheduler设定
        if 'scheduler' in kwargs:
            if kwargs['scheduler'] == 'step_lr':
                # step_size设置，默认为50
                self.step_size = kwargs['step_size'] if 'step_lr' in kwargs else 100
                # gamma设置，默认为0.5
                self.gamma = kwargs['gamma'] if 'gamma' in kwargs else 0.5

                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.step_size, self.gamma)
        else:
            self.scheduler = None

        # 训练参数设定
        self.epochs = kwargs['epochs'] if 'epochs' in kwargs else 100                       # epochs默认100
        if 'checkpoint_dir' in kwargs:                                                      # checkpoint
            self.checkpoint_dir = kwargs['checkpoint_dir']
        else:
            self.checkpoint_dir = os.path.join('./checkpoint', kwargs['model']+'_'+kwargs['criterion'])
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.model_dir = kwargs['model_dir'] if 'model_dir' in kwargs else './models/model.pth'             # model引继点设定

        # 训练数据集设定
        if 'train_datasets' in kwargs:
            print('载入训练集')
            self.train_datasets = kwargs['train_datasets']()
            if type(self.train_datasets) is not data.DataLoader:
                print('错误：不支持的datasets类型，详情见Readme。')
                exit(4)
        else:
            print('错误：Train缺少训练数据集，例如Train(train_datasets=...)，详细规则见Readme。')
            exit(2)

        # 验证数据集设定
        if 'valid_datasets' in kwargs:
            print('载入验证集')
            self.valid_datasets = kwargs['valid_datasets']()
            if type(self.valid_datasets) is not data.DataLoader:
                print('错误：不支持的datasets类型，详情见Readme。')
                exit(4)
        else:
            self.valid_datasets = None

        self.loss_arr = {'train_loss': [], 'valid_loss': []}


    def train(self):
        print('device: ', self.device)
        if (self.model_dir is not None) and (os.path.exists(self.model_dir)):  # 载入原有模型继续训练
            self.model.load_state_dict(torch.load(self.model_dir))
        min_loss = 1
        best_epoch = 1

        # Epoch
        for epoch in range(self.epochs):
            self.model.train()  # 开启训练模式
            print('Epoch: %d/%d, lr: %.6f' % (epoch + 1, self.epochs,
                                              self.optimizer.state_dict()['param_groups'][0]['lr']))
            epoch_loss = 0
            step = 0
            start_time = time.time()
            is_save_checkpoint = False

            # 训练网络
            for _, (t_x, t_y) in enumerate(self.train_datasets):
                step += 1

                # 训练batch放入显存
                t_x = t_x.to(self.device)
                t_y = t_y.to(self.device)
                # 用单个Batch训练网络
                out = self.model(t_x)
                loss = self.criterion(out, t_y)
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

            epoch_loss = epoch_loss / step
            self.loss_arr['train_loss'].append(epoch_loss)

            if self.valid_datasets:
                val_loss = self.validate()
                print('time: ', int(time.time() - start_time), 'train_loss: ', epoch_loss, 'valid_loss: ', val_loss)
            else:
                print('time: ', int(time.time() - start_time), 'train_loss: ', epoch_loss)
            if min_loss > epoch_loss:
                min_loss = epoch_loss
                is_save_checkpoint = True

            if is_save_checkpoint:
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, str(epoch + 1) + '.pth'))

        print('Best Epoch: ', best_epoch)
        shutil.copy(os.path.join(self.checkpoint_dir, str(best_epoch) + '.pth'), self.model_dir)

    def validate(self) -> float:
        """
        验证过程
        :return: 验证集loss
        """
        step = 0
        valid_loss = 0
        with torch.no_grad():
            self.model.eval()
            for _, (v_x, v_y) in enumerate(self.valid_datasets):
                step += 1

                v_x = v_x.to(self.device)
                v_y = v_y.to(self.device)
                out = self.model(v_x)
                loss = self.criterion(out, v_y)
                valid_loss += loss.item()

            valid_loss = valid_loss / step
            self.loss_arr['valid_loss'].append(valid_loss)
        torch.cuda.empty_cache()
        return valid_loss

    def show_loss_arr(self):
        print('train loss: \n', self.loss_arr['train_loss'])
        print('valid loss: \n', self.loss_arr['valid_loss'])
