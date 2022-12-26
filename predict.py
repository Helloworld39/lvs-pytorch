import os.path
import torch
import torch.nn as nn
import data
import evaluation as ev
from model import UNet2D, UNet3D
from torchvision.transforms import transforms


class Predict:
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
                self.in_channels = kwargs['in_channels'] if 'in_channels' in kwargs else 1
                self.out_channels = kwargs['out_channels'] if 'out_channels' in kwargs else 1
                self.model = UNet2D(self.in_channels, self.out_channels).to(self.device)
            elif kwargs['model'] == 'unet3d':
                self.in_channels = kwargs['in_channels'] if 'in_channels' in kwargs else 1
                self.out_channels = kwargs['out_channels'] if 'out_channels' in kwargs else 1
                self.model = UNet3D(self.in_channels, self.out_channels).to(self.device)
            else:
                print('错误：不存在的模型。')
                exit(3)
        else:
            print('错误：Predict中缺少必要参数[model]，例如Predict(model="unet", ...)')
            exit(2)

        # criterion设定
        if 'criterion' in kwargs:
            if kwargs['criterion'] == 'bce':
                self.criterion = nn.BCELoss().to(self.device)
            elif kwargs['criterion'] == 'ce':
                self.criterion = nn.CrossEntropyLoss().to(self.device)
            elif kwargs['criterion'] == 'mse':
                self.criterion = nn.MSELoss().to(self.device)
            else:
                print('错误：不存在的损失函数。')
                exit(3)
        else:
            print('错误：Predict中缺少必要参数[criterion]，例如Predict(criterion="bce", ...)')
            exit(2)

        if 'model_dir' in kwargs:
            self.model_dir = kwargs['model_dir']
        else:
            print('错误：Predict中缺少必要参数[model_dir]，例如Predict(model_dir=...)')
            exit(2)

        self.output_dir = kwargs['output_dir'] if 'output_dir' in kwargs else './out'
        self.output_index = kwargs['output_index'] if 'output_index' in kwargs else 1
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if 'pre_datasets' in kwargs:
            print('载入测试集')
            self.pre_datasets = kwargs['pre_datasets']()
            if type(self.pre_datasets) is not data.DataLoader:
                print('错误：不支持的datasets类型，详情见Readme。')
                exit(4)
        else:
            print('错误：Predict缺少测试数据集，例如Predict(pre_datasets=...)，详细规则见Readme。')
            exit(2)

    def predict(self):
        print('device: ', self.device)
        step = 0
        test_loss = 0
        test_dice = 0
        self.model.load_state_dict(torch.load(self.model_dir))
        with torch.no_grad():
            self.model.eval()
            for _, (t_x, t_y) in enumerate(self.pre_datasets):
                step += 1

                t_x = t_x.to(self.device)
                t_y = t_y.to(self.device)

                out = self.model(t_x)

                loss = self.criterion(out, t_y)
                test_loss += loss.item()

                d_score = ev.dice_score(out, t_y)
                test_dice += d_score

                self.save_result(out)

        test_loss = test_loss / step
        test_dice = test_dice / step
        print('Test Loss: ', test_loss)

    def save_result(self, mat: torch.Tensor):
        output_index = self.output_index
        output_dir = self.output_dir
        to_image = transforms.ToPILImage()
        for i in range(mat.shape[0]):
            print(type(mat[i]))
            image = to_image(mat[i])
            image.save(os.path.join(output_dir, str(output_index)+'.png'))
            output_index += 1
        self.output_index = output_index
