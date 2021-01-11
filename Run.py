import torch
import sys
# sys.path.append("..")

from NNTrain import trainModels
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ====================================

if __name__ == '__main__':

    trainModels(data_directory='./datasets/',
                dataset_name='debug',
                input_dim=3,
                class_no=2,
                repeat=1,
                train_batchsize=4,
                validate_batchsize=1,
                num_epochs=10,
                learning_rate=1e-4,
                width=32,
                network='unet',
                augmentation='all_flip',
                reverse=False)
