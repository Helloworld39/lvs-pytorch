import data
from train import Train
from predict import Predict


dataset_root_dir = data.dataset_dir_manager('msdc', is_root=True)
train_dataset = data.data_loader(dataset_root_dir+'/msdc_5d_train.pth', 8, True)
valid_dataset = data.data_loader(dataset_root_dir+'/msdc_5d_valid.pth', 8)

train = Train(model='unet3d',
              criterion='bce',
              optimizer='adam', lr=1e-4,
              scheduler='step_lr',
              epochs=100,
              checkpoint_dir=dataset_root_dir+'checkpoint/unet3d_msdc_100_8',
              model_dir='./models/unet3d_msdc_100_8.pth',
              train_datasets=train_dataset,
              valid_datasets=valid_dataset)
train.train()
train.show_loss_arr()

test_dataset = data.data_loader(dataset_root_dir+'/msdc_5d_test.pth', 8)
predict = Predict(model='unet3d',
                  criterion='bce',
                  model_dir='./models/unet3d_msdc_100_8.pth',
                  output_dir='/root/autodl-tmp/msdc/out/unet3d_msdc_100_8',
                  output_index=data.get_ct_index('msdc')[298],
                  pre_datasets=test_dataset)
predict.predict()
