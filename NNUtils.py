import torch
import errno
import numpy as np
import os
# import Image
import torch.nn as nn
import glob
import tifffile as tiff

import random

from adamW import AdamW
from NNMetrics import segmentation_scores, f1_score, hd95, preprocessing_accuracy
from PIL import Image
from torch.utils import data

from NNLoss import dice_loss
# =============================================


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, imgs_folder, labels_folder, augmentation):

        # 1. Initialize file paths or a list of file names.
        self.imgs_folder = imgs_folder
        self.labels_folder = labels_folder
        self.data_augmentation = augmentation
        # self.transform = transforms

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using num py.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).

        all_images = glob.glob(os.path.join(self.imgs_folder, '*.tif'))
        all_labels = glob.glob(os.path.join(self.labels_folder, '*.tif'))
        # sort all in the same order
        all_labels.sort()
        all_images.sort()
        #
        # label = Image.open(all_labels[index])
        label = tiff.imread(all_labels[index])
        label_origin = np.array(label, dtype='float32')
        image = tiff.imread(all_images[index])
        image = np.array(image, dtype='float32')
        #
        labelname = all_labels[index]
        path_label, labelname = os.path.split(labelname)
        labelname, labelext = os.path.splitext(labelname)
        #
        c_amount = len(np.shape(label))
        # Reshaping everyting to make sure the order: channel x height x width
        if c_amount == 3:
            #
            d1, d2, d3 = np.shape(label)
            #
            if d1 != min(d1, d2, d3):
                #
                # label = np.reshape(label, (d3, d1, d2))
                label = np.transpose(label_origin, (2, 0, 1))
                c = d3
                h = d1
                w = d2
            else:
                c = d1
                h = d2
                w = d3
            #
        elif c_amount == 2:
            h, w = np.shape(label)
            # label = np.reshape(label_origin, (1, h, w))
            label = np.expand_dims(label_origin, axis=0)
        #
        c_amount = len(np.shape(image))
        #
        if c_amount == 3:
            #
            d1, d2, d3 = np.shape(image)
            #
            if d1 != min(d1, d2, d3):
                #
                # image = np.reshape(image, (d3, d1, d2))
                image = np.transpose(image, (2, 0, 1))
                #
        elif c_amount == 2:
            #
            image = np.expand_dims(image, axis=0)
        #
        if self.data_augmentation == 'full':
            # augmentation:
            augmentation = random.uniform(0, 1)
            #
            if augmentation < 0.2:
                #
                c, h, w = np.shape(image)
                #
                for channel in range(c):
                    #
                    image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                    image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
                    #
                label = np.flip(label, axis=1).copy()
                label = np.flip(label, axis=2).copy()

            elif augmentation < 0.4:
                #
                c, h, w = np.shape(image)
                #
                for channel in range(c):
                    #
                    image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                    #
                label = np.flip(label, axis=1).copy()

            # elif augmentation < 0.375:
            #     #
            #     c, h, w = np.shape(image)
            #     #
            #     for channel in range(c):
            #         #
            #         image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
            #         #
            #     label = np.flip(label, axis=2).copy()

            elif augmentation < 0.6:
                #
                mean = 0.0
                sigma = 0.15
                noise = np.random.normal(mean, sigma, image.shape)
                mask_overflow_upper = image + noise >= 1.0
                mask_overflow_lower = image + noise < 0.0
                noise[mask_overflow_upper] = 1.0
                noise[mask_overflow_lower] = 0.0
                image += noise

            # elif augmentation < 0.625:
            #     #
            #     c, h, w = np.shape(image)
            #     #
            #     for channel in range(c):
            #         #
            #         channel_ratio = random.uniform(0, 1)
            #         #
            #         image[channel, :, :] = image[channel, :, :] * channel_ratio

            # elif augmentation < 0.75:
            #     #
            #     c, h, w = np.shape(image)
            #     #
            #     for channel in range(c):
            #         #
            #         channel_ratio = random.uniform(0, 1)
            #         #
            #         image[channel, :, :] = image[channel, :, :] * channel_ratio
            #         image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
            #         image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
            #         #
            #     label = np.flip(label, axis=1).copy()
            #     label = np.flip(label, axis=2).copy()

            elif augmentation < 0.8:
                #
                c, h, w = np.shape(image)
                #
                for channel in range(c):
                    #
                    # channel_ratio = random.uniform(0, 1)
                    # image[channel, :, :] = image[channel, :, :] * channel_ratio
                    image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                    image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
                    #
                label = np.flip(label, axis=1).copy()
                label = np.flip(label, axis=2).copy()
                #
                mean = 0.0
                sigma = 0.15
                noise = np.random.normal(mean, sigma, image.shape)
                mask_overflow_upper = image + noise >= 1.0
                mask_overflow_lower = image + noise < 0.0
                noise[mask_overflow_upper] = 1.0
                noise[mask_overflow_lower] = 0.0
                image += noise

        elif self.data_augmentation == 'flip':
            # augmentation:
            augmentation = random.uniform(0, 1)
            #
            if augmentation > 0.5 or augmentation == 0.5:
                #
                c, h, w = np.shape(image)
                #
                for channel in range(c):
                    #
                    image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                    #
                label = np.flip(label, axis=1).copy()

        elif self.data_augmentation == 'all_flip':
            # augmentation:
            augmentation = random.uniform(0, 1)

            if augmentation <= 0.25:

                c, h, w = np.shape(image)
                for channel in range(c):
                    image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                    image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()

                label = np.flip(label, axis=1).copy()
                label = np.flip(label, axis=2).copy()

            elif augmentation <= 0.5:

                c, h, w = np.shape(image)
                for channel in range(c):
                    image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                label = np.flip(label, axis=1).copy()

            elif augmentation <= 0.75:

                c, h, w = np.shape(image)
                for channel in range(c):
                    image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
                label = np.flip(label, axis=2).copy()

            else:

                label = label
                image = image

        elif self.data_augmentation == 'Gaussian':

            mean = 0.0
            sigma = 0.15
            noise = np.random.normal(mean, sigma, image.shape)
            mask_overflow_upper = image + noise >= 1.0
            mask_overflow_lower = image + noise < 0.0
            noise[mask_overflow_upper] = 1.0
            noise[mask_overflow_lower] = 0.0
            image += noise

        else:

            label = label
            image = image

        c, h, w = np.shape(image)

        # if h == 512 or w == 512:
        #     #
        #     image = image[:, 0::2, 0::2]
        #     label = label[:, 0::2, 0::2]
        #     #
        # elif h == 224 or w == 224:
        #
        #     image = image[:, 0::2, 0::2]
        #     label = label[:, 0::2, 0::2]
            #
        return image, label, labelname

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(glob.glob(os.path.join(self.imgs_folder, '*.tif')))


# ============================================================================================


def evaluate(evaluatedata, model, device, reverse_mode, class_no):

    model.eval()

    f1 = 0
    test_iou = 0
    test_h_dist = 0
    recall = 0
    precision = 0

    FPs_Ns = 0
    FNs_Ps = 0
    FPs_Ps = 0
    FNs_Ns = 0
    TPs = 0
    TNs = 0
    FNs = 0
    FPs = 0
    Ps = 0
    Ns = 0

    effective_h = 0

    for j, (testimg, testlabel, testname) in enumerate(evaluatedata):
        # validate batch size will be set up as 2
        # j will be close enough to the

        testimg = testimg.to(device=device, dtype=torch.float32)
        testimg.requires_grad = True

        testlabel = testlabel.to(device=device, dtype=torch.float32)

        threshold = torch.tensor([0.5], dtype=torch.float32, device=device, requires_grad=False)

        upper = torch.tensor([1.0], dtype=torch.float32, device=device, requires_grad=False)

        lower = torch.tensor([0.0], dtype=torch.float32, device=device, requires_grad=False)

        testoutput = model(testimg)

        prob_testoutput = torch.sigmoid(testoutput)

        # attack testing data:
        loss = dice_loss(prob_testoutput, testlabel)
        model.zero_grad()
        loss.backward()
        data_grad = testimg.grad.data
        perturbed_data = fgsm_attack(testimg, 0.2, data_grad)
        prob_testoutput = model(perturbed_data)
        prob_testoutput = torch.sigmoid(prob_testoutput)

        if reverse_mode is True:

            testoutput = torch.where(prob_testoutput < threshold, upper, lower)

        else:

            testoutput = torch.where(prob_testoutput > threshold, upper, lower)

        mean_iu_ = segmentation_scores(testlabel, testoutput, class_no)

        if (testoutput == 1).sum() > 1 and (testlabel == 1).sum() > 1:

            h_dis95_ = hd95(testoutput, testlabel, class_no)
            test_h_dist += h_dis95_
            effective_h = effective_h + 1

        f1_, recall_, precision_, TP, TN, FP, FN, P, N = f1_score(testlabel, testoutput, class_no)

        f1 += f1_
        test_iou += mean_iu_
        recall += recall_
        precision += precision_
        TPs += TP
        TNs += TN
        FPs += FP
        FNs += FN
        Ps += P
        Ns += N
        FNs_Ps += (FN + 1e-10) / (P + 1e-10)
        FPs_Ns += (FP + 1e-10) / (N + 1e-10)
        FNs_Ns += (FN + 1e-10) / (N + 1e-10)
        FPs_Ps += (FP + 1e-10) / (P + 1e-10)

    return test_iou / (j+1), f1 / (j+1), recall / (j+1), precision / (j+1), FPs_Ns / (j+1), FPs_Ps / (j+1), FNs_Ns / (j+1), FNs_Ps / (j+1), FPs / (j+1), FNs / (j+1), TPs / (j+1), TNs / (j+1), Ps / (j+1), Ns / (j+1), test_h_dist / (effective_h + 1)


def test(
        testdata,
         models_path,
         device,
         reverse_mode,
         class_no,
         save_path):

    all_models = glob.glob(os.path.join(models_path, '*.pt'))

    test_f1 = []
    test_iou = []
    test_h_dist = []
    test_recall = []
    test_precision = []

    test_iou_adv = []
    test_h_dist_adv = []

    for model in all_models:

        model = torch.load(model)
        model.eval()

        for j, (testimg, testlabel, testname) in enumerate(testdata):
            # validate batch size will be set up as 2
            # testimg = torch.from_numpy(testimg).to(device=device, dtype=torch.float32)
            # testlabel = torch.from_numpy(testlabel).to(device=device, dtype=torch.float32)

            testimg = testimg.to(device=device, dtype=torch.float32)
            testimg.requires_grad = True

            testlabel = testlabel.to(device=device, dtype=torch.float32)

            threshold = torch.tensor([0.5], dtype=torch.float32, device=device, requires_grad=False)
            upper = torch.tensor([1.0], dtype=torch.float32, device=device, requires_grad=False)
            lower = torch.tensor([0.0], dtype=torch.float32, device=device, requires_grad=False)

            # c, h, w = testimg.size()
            # testimg = testimg.expand(1, c, h, w)

            testoutput = model(testimg)
            # (todo) add for multi-class
            prob_testoutput = torch.sigmoid(testoutput)

            if class_no == 2:

                if reverse_mode is True:

                    testoutput = torch.where(prob_testoutput < threshold, upper, lower)

                else:

                    testoutput = torch.where(prob_testoutput > threshold, upper, lower)

            # metrics before attack:
            mean_iu_ = segmentation_scores(testlabel, testoutput, class_no)
            test_iou.append(mean_iu_)

            if (testoutput == 1).sum() > 1 and (testlabel == 1).sum() > 1:
                h_dis95_ = hd95(testoutput, testlabel, class_no)
                test_h_dist.append(h_dis95_)

            f1_, recall_, precision_, TP, TN, FP, FN, P, N = f1_score(testlabel, testoutput, class_no)
            test_f1.append(f1_)
            test_recall.append(recall_)
            test_precision.append(precision_)

            # attack testing data:
            loss = dice_loss(prob_testoutput, testlabel)
            model.zero_grad()
            loss.backward()
            data_grad = testimg.grad.data
            perturbed_data = fgsm_attack(testimg, 0.2, data_grad)
            prob_testoutput_adv = model(perturbed_data)
            prob_testoutput_adv = torch.sigmoid(prob_testoutput_adv)

            if class_no == 2:

                if reverse_mode is True:

                    testoutput_adv = torch.where(prob_testoutput_adv < threshold, upper, lower)

                else:

                    testoutput_adv = torch.where(prob_testoutput_adv > threshold, upper, lower)

            mean_iu_adv_ = segmentation_scores(testlabel, testoutput_adv, class_no)
            test_iou_adv.append(mean_iu_adv_)

            if (testoutput_adv == 1).sum() > 1 and (testlabel == 1).sum() > 1:
                h_dis95_adv_ = hd95(testoutput_adv, testlabel, class_no)
                test_h_dist_adv.append(h_dis95_adv_)

    # store the test metrics

    prediction_map_path = save_path + '/Test_result'

    try:

        os.mkdir(prediction_map_path)

    except OSError as exc:

        if exc.errno != errno.EEXIST:

            raise

        pass

    # save numerical results:
    result_dictionary = {
        'Test IoU mean': str(np.mean(test_iou)),
        'Test IoU std': str(np.std(test_iou)),
        'Test f1 mean': str(np.mean(test_f1)),
        'Test f1 std': str(np.std(test_f1)),
        'Test H-dist mean': str(np.mean(test_h_dist)),
        'Test H-dist std': str(np.std(test_h_dist)),
        'Test precision mean': str(np.mean(test_precision)),
        'Test precision std': str(np.std(test_precision)),
        'Test recall mean': str(np.mean(test_recall)),
        'Test recall std': str(np.std(test_recall)),
        'Test IoU attack mean': str(np.mean(test_iou_adv)),
        'Test IoU attack std': str(np.std(test_iou_adv)),
        'Test H-dist attack mean': str(np.mean(test_h_dist_adv)),
        'Test H-dist attack std': str(np.std(test_h_dist_adv)),
    }

    ff_path = prediction_map_path + '/test_results.txt'
    ff = open(ff_path, 'w')
    ff.write(str(result_dictionary))
    ff.close()

    print(
        'Test h-dist: {:.4f}, '
        'Val iou: {:.4f}, '.format(np.mean(test_h_dist), np.mean(test_iou)))

# ==============================================================================================


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# ========================================

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
        (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size,
                       kernel_size), dtype=np.float32)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()
