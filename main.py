import data
from train import Train
from predict import Predict


dataset_name = 'msdc'
need_create_dataset = True

slice_index_list = data.get_ct_index(dataset_name)
dataset_root_dir = data.dataset_dir_manager(dataset_name, is_root=True)
data_dir = data.dataset_dir_manager(dataset_name)

if need_create_dataset:
    data.create_4d_tensor_dataset(dataset_root_dir+'/train.pth', data_dir,
                                  slice_index_list[0], slice_index_list[101], input_type=2)
    data.create_4d_tensor_dataset(dataset_root_dir+'/valid.pth', data_dir,
                                  slice_index_list[101], slice_index_list[103], input_type=2)
    data.create_4d_tensor_dataset(dataset_root_dir+'/predict.pth', data_dir,
                                  slice_index_list[298], slice_index_list[303], input_type=2)

train_dataset = data.data_loader(dataset_root_dir+'/train.pth', 16, True)
valid_dataset = data.data_loader(dataset_root_dir+'/valid.pth', 16)

train = Train(model='2in1out',
              criterion='bce',
              optimizer='adam', lr=1e-3,
              scheduler='step_lr', gamma=0.5,
              epochs=150,
              checkpoint_dir=dataset_root_dir+'/checkpoint/2in1out_msdc_150_pj',
              model_dir='./models/2in1out_msdc_150_pj.pth',
              train_datasets=train_dataset,
              valid_datasets=valid_dataset)
train.train()
train.show_loss_arr()

test_dataset = data.data_loader(dataset_root_dir+'/predict.pth', 16)
predict = Predict(model='2in1out',
                  criterion='bce',
                  model_dir='./models/2in1out_msdc_150_pj.pth',
                  output_dir=dataset_root_dir+'/out/2in1out_msdc_150_pj',
                  output_index=slice_index_list[298],
                  pre_datasets=test_dataset)
predict.predict()
