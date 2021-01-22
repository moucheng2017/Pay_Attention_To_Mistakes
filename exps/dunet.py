import torch
import sys
sys.path.append("..")
from NNTrain import trainModels
# torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    trainModels(data_directory='/SAN/medic/PerceptronHead/data/brain/BRATS2018/',
                dataset_name='ET_L0_H50',
                input_dim=4,
                class_no=2,
                repeat=1,
                train_batchsize=4,
                validate_batchsize=1,
                num_epochs=100,
                learning_rate=1e-2,
                width=32,
                network='dilated_unet',
                dilation=9,
                augmentation='all_flip',
                reverse=False)

