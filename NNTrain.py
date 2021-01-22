import os
import errno
import torch
import timeit

import glob

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.functional as F
from torch.utils import data

# import Image
# from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances
from scipy import spatial
from sklearn.metrics import mean_squared_error
from torch.optim import lr_scheduler
from NNLoss import dice_loss
from NNMetrics import segmentation_scores, f1_score, hd95
from NNUtils import CustomDataset, evaluate, test
from tensorboardX import SummaryWriter
from adamW import AdamW
from torch.autograd import Variable

from Model import ERFANet
from NNBaselines import GCNonLocal_UNet_All
from NNBaselines import UNet
from NNBaselines import CBAM_UNet_All
from NNBaselines import DilatedUNet
from NNBaselines import AttentionUNet
from NNBaselines import CSE_UNet_Full

from NNUtils import fgsm_attack


def trainModels(
                data_directory,
                dataset_name,
                input_dim,
                class_no,
                repeat,
                train_batchsize,
                validate_batchsize,
                num_epochs,
                learning_rate,
                width,
                network,
                dilation,
                lr_decay=True,
                augmentation=True,
                reverse=False):

    for j in range(1, repeat + 1):

        repeat_str = str(j)

        if 'ERF' in network or 'erf' in network:

            assert 'fp' in network or 'fn' in network or 'FP' in network or 'FN' in network
            assert 'encoder' in network or 'decoder' in network or 'all' in network

            if 'fp' in network or 'FP' in network:
                attention_type = 'FP'
                assert reverse is False
            elif 'fn' in network or 'FN' in network:
                attention_type = 'FN'
                assert reverse is True
            else:
                attention_type = 'FP'
                assert reverse is False

            if 'encoder' in network:
                mode = 'encoder'
            elif 'decoder' in network:
                mode = 'decoder'
            elif 'all' in network:
                mode = 'all'
            else:
                mode = 'all'

            Exp = ERFANet(
                in_ch=input_dim,
                width=width,
                class_no=class_no,
                attention_type=attention_type,
                mode=mode,
                identity_add=True,
                dilation=dilation)

            if 'fp' in network:

                Exp_name = network + \
                           '_batch_' + str(train_batchsize) + \
                           '_width_' + str(width) + \
                           '_repeat_' + repeat_str + \
                           '_augment_' + str(augmentation) + \
                           '_lr_decay_' + str(lr_decay) + '_dilation_' + str(dilation)

            else:

                Exp_name = network + \
                           '_batch_' + str(train_batchsize) + \
                           '_width_' + str(width) + \
                           '_repeat_' + repeat_str + \
                           '_augment_' + str(augmentation) + \
                           '_lr_decay_' + str(lr_decay)

        # ==================================================
        # Baselines
        # ==================================================

        elif network == 'unet':
            assert reverse is False
            Exp = UNet(in_ch=input_dim, width=width, class_no=class_no)
            Exp_name = 'UNet_batch_' + str(train_batchsize) + \
                       '_width_' + str(width) + \
                       '_repeat_' + repeat_str + \
                       '_augment_' + str(augmentation) + \
                       '_lr_decay_' + str(lr_decay)

        elif network == 'dilated_unet':
            assert reverse is False
            # dilation = 9
            Exp = DilatedUNet(in_ch=input_dim, width=width, dilation=dilation)
            Exp_name = 'DilatedUNet_batch_' + str(train_batchsize) + \
                       '_width_' + str(width) + \
                       '_dilation_' + str(dilation) + \
                       '_repeat_' + repeat_str + \
                       '_augment_' + str(augmentation) + \
                       '_lr_decay_' + str(lr_decay)

        elif network == 'atten_unet':
            assert reverse is False
            Exp = AttentionUNet(in_ch=input_dim, width=width)
            Exp_name = 'AttentionUNet_batch_' + str(train_batchsize) + \
                       '_Valbatch_' + str(validate_batchsize) + \
                       '_width_' + str(width) + \
                       '_repeat_' + repeat_str + \
                       '_augment_' + str(augmentation) + \
                       '_lr_decay_' + str(lr_decay)

        elif network == 'cse_unet_full':
            # assert visualise_attention is True
            assert reverse is False
            # didn't have time to write the code to visulisae attention weights for cse u net
            Exp = CSE_UNet_Full(in_ch=input_dim, width=width)
            Exp_name = 'CSEUNetFull_batch_' + str(train_batchsize) + \
                       '_Valbatch_' + str(validate_batchsize) + \
                       '_width_' + str(width) + \
                       '_repeat_' + repeat_str + \
                       '_augment_' + str(augmentation) + \
                       '_lr_decay_' + str(lr_decay)

        # ====================================================================================================================================================================
        trainloader, validateloader, testloader, train_dataset, validate_dataset, test_dataset = getData(data_directory, dataset_name, train_batchsize, validate_batchsize, augmentation)
        # ===================
        trainSingleModel(Exp,
                         Exp_name,
                         num_epochs,
                         learning_rate,
                         dataset_name,
                         train_dataset,
                         train_batchsize,
                         trainloader,
                         validateloader,
                         testloader,
                         reverse_mode=reverse,
                         lr_schedule=lr_decay,
                         class_no=class_no)


def getData(data_directory, dataset_name, train_batchsize, validate_batchsize, data_augment):

    train_image_folder = data_directory + dataset_name + '/train/patches'
    train_label_folder = data_directory + dataset_name + '/train/labels'
    validate_image_folder = data_directory + dataset_name + '/validate/patches'
    validate_label_folder = data_directory + dataset_name + '/validate/labels'
    test_image_folder = data_directory + dataset_name + '/test/patches'
    test_label_folder = data_directory + dataset_name + '/test/labels'

    train_dataset = CustomDataset(train_image_folder, train_label_folder, data_augment)

    validate_dataset = CustomDataset(validate_image_folder, validate_label_folder, 'full')

    test_dataset = CustomDataset(test_image_folder, test_label_folder, 'full')

    trainloader = data.DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=4, drop_last=True)

    validateloader = data.DataLoader(validate_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    return trainloader, validateloader, testloader, train_dataset, validate_dataset, test_dataset

# =====================================================================================================================================


def trainSingleModel(model,
                     model_name,
                     num_epochs,
                     learning_rate,
                     datasettag,
                     train_dataset,
                     train_batchsize,
                     trainloader,
                     validateloader,
                     testdata,
                     reverse_mode,
                     lr_schedule,
                     class_no):

    # change log names
    training_amount = len(train_dataset)

    iteration_amount = training_amount // train_batchsize

    iteration_amount = iteration_amount - 1

    device = torch.device('cuda')

    lr_str = str(learning_rate)

    epoches_str = str(num_epochs)

    save_model_name = model_name + '_' + datasettag + '_e' + epoches_str + '_lr' + lr_str

    saved_information_path = './Results'
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    saved_information_path = saved_information_path + '/' + save_model_name
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    saved_model_path = saved_information_path + '/trained_models'
    try:
        os.mkdir(saved_model_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    print('The current model is:')
    print(save_model_name)
    print('\n')

    writer = SummaryWriter('./Results/Log_' + datasettag + '/' + save_model_name)

    model.to(device)

    threshold = torch.tensor([0.5], dtype=torch.float32, device=device, requires_grad=False)
    upper = torch.tensor([1.0], dtype=torch.float32, device=device, requires_grad=False)
    lower = torch.tensor([0.0], dtype=torch.float32, device=device, requires_grad=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    if lr_schedule is True:

        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=10, threshold=0.001)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[num_epochs // 2, 3*num_epochs // 4], gamma=0.1)

    start = timeit.default_timer()

    for epoch in range(num_epochs):

        model.train()

        h_dists = 0
        f1 = 0
        accuracy_iou = 0
        running_loss = 0
        recall = 0
        precision = 0

        t_FPs_Ns = 0
        t_FPs_Ps = 0
        t_FNs_Ns = 0
        t_FNs_Ps = 0
        t_FPs = 0
        t_FNs = 0
        t_TPs = 0
        t_TNs = 0
        t_Ps = 0
        t_Ns = 0

        effective_h = 0

        # j: index of iteration
        for j, (images, labels, imagename) in enumerate(trainloader):

            # check training data:
            # image = images[0, :, :, :].squeeze().detach().cpu().numpy()
            # label = labels[0, :, :, :].squeeze().detach().cpu().numpy()
            # image = np.transpose(image, (1, 2, 0))
            # label = np.expand_dims(label, axis=2)
            # label = np.concatenate((label, label, label), axis=2)
            # plt.imshow(0.5*image + 0.5*label)
            # plt.show()

            optimizer.zero_grad()

            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)

            images.requires_grad = True

            if reverse_mode is True:

                inverse_labels = torch.ones_like(labels)
                inverse_labels = inverse_labels.to(device=device, dtype=torch.float32)
                inverse_labels = inverse_labels - labels
            else:
                pass

            outputs = model(images)
            prob_outputs = torch.sigmoid(outputs)

            if reverse_mode is True:
                loss = dice_loss(prob_outputs, inverse_labels)
            else:
                loss = dice_loss(prob_outputs, labels)

            loss.backward()
            optimizer.step()

            # The taks of binary segmentation is too easy, to compensate the simplicity of the task,
            # we add adversarial noises in the testing images:
            data_grad = images.grad.data
            perturbed_data = fgsm_attack(images, 0.2, data_grad)
            prob_outputs = model(perturbed_data)
            prob_outputs = torch.sigmoid(prob_outputs)

            if reverse_mode is True:
                class_outputs = torch.where(prob_outputs < threshold, upper, lower)
            else:
                class_outputs = torch.where(prob_outputs > threshold, upper, lower)

            if class_no == 2:
                # hasudorff distance is for binary
                if (class_outputs == 1).sum() > 1 and (labels == 1).sum() > 1:
                    dist_ = hd95(class_outputs, labels, class_no)
                    h_dists += dist_
                    effective_h = effective_h + 1
                else:
                    pass
            else:
                pass

            mean_iu_ = segmentation_scores(labels, class_outputs, class_no)
            f1_, recall_, precision_, TPs_, TNs_, FPs_, FNs_, Ps_, Ns_ = f1_score(labels, class_outputs, class_no)

            running_loss += loss
            f1 += f1_
            accuracy_iou += mean_iu_
            recall += recall_
            precision += precision_
            t_TPs += TPs_
            t_TNs += TNs_
            t_FPs += FPs_
            t_FNs += FNs_
            t_Ps += Ps_
            t_Ns += Ns_
            t_FNs_Ps += (FNs_ + 1e-8) / (Ps_ + 1e-8)
            t_FPs_Ns += (FPs_ + 1e-8) / (Ns_ + 1e-8)
            t_FNs_Ns += (FNs_ + 1e-8) / (Ns_ + 1e-8)
            t_FPs_Ps += (FPs_ + 1e-8) / (Ps_ + 1e-8)

            if (j + 1) % iteration_amount == 0:

                validate_iou, validate_f1, validate_recall, validate_precision, v_FPs_Ns, v_FPs_Ps, v_FNs_Ns, v_FNs_Ps, v_FPs, v_FNs, v_TPs, v_TNs, v_Ps, v_Ns, v_h_dist = evaluate(validateloader, model, device, reverse_mode=reverse_mode, class_no=class_no)

                print(
                    'Step [{}/{}], Train loss: {:.4f}, '
                    'Train iou: {:.4f}, '
                    'Train h-dist:{:.4f}, '
                    'Val iou: {:.4f},'
                    'Val h-dist: {:.4f}'.format(epoch + 1, num_epochs,
                                                   running_loss / (j + 1),
                                                   accuracy_iou / (j + 1),
                                                   h_dists / (effective_h + 1),
                                                   validate_iou,
                                                   v_h_dist))

                # # # ================================================================== #
                # # #                        TensorboardX Logging                        #
                # # # # ================================================================ #

                writer.add_scalars('acc metrics', {'train iou': accuracy_iou / (j+1),
                                                   'train hausdorff dist': h_dists / (effective_h+1),
                                                   'val iou': validate_iou,
                                                   'val hasudorff distance': v_h_dist,
                                                   'loss': running_loss / (j+1)}, epoch + 1)

                writer.add_scalars('train confusion matrices analysis', {'train FPs/Ns': t_FPs_Ns / (j+1),
                                                                         'train FNs/Ps': t_FNs_Ps / (j+1),
                                                                         'train FPs/Ps': t_FPs_Ps / (j+1),
                                                                         'train FNs/Ns': t_FNs_Ns / (j+1),
                                                                         'train FNs': t_FNs / (j+1),
                                                                         'train FPs': t_FPs / (j+1),
                                                                         'train TNs': t_TNs / (j+1),
                                                                         'train TPs': t_TPs / (j+1),
                                                                         'train Ns': t_Ns / (j+1),
                                                                         'train Ps': t_Ps / (j+1),
                                                                         'train imbalance': t_Ps / (t_Ps + t_Ns)}, epoch + 1)

                writer.add_scalars('val confusion matrices analysis', {'val FPs/Ns': v_FPs_Ns,
                                                                       'val FNs/Ps': v_FNs_Ps,
                                                                       'val FPs/Ps': v_FPs_Ps,
                                                                       'val FNs/Ns': v_FNs_Ns,
                                                                       'val FNs': v_FNs,
                                                                       'val FPs': v_FPs,
                                                                       'val TNs': v_TNs,
                                                                       'val TPs': v_TPs,
                                                                       'val Ns': v_Ns,
                                                                       'val Ps': v_Ps,
                                                                       'val imbalance': v_Ps / (v_Ps + v_Ns)}, epoch + 1)
            else:
                pass

            # A learning rate schedule plan for fn attention:
            # we ramp-up linearly inside of each iteration
            # without the warm-up, it is hard to train sometimes
            if 'fn' in model_name or 'FN' in model_name:
                if reverse_mode is True:
                    if epoch < 10:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate * (j / len(trainloader))
                    else:
                        pass
                else:
                    pass
            else:
                pass

        if lr_schedule is True:
            scheduler.step()
        else:
            pass

        # save models at last 10 epochs
        if epoch >= (num_epochs - 10):
            save_model_name_full = saved_model_path + '/' + save_model_name + '_epoch' + str(epoch) + '.pt'
            path_model = save_model_name_full
            torch.save(model, path_model)

    # Test on all models and average them:
    test(testdata,
         saved_model_path,
         device,
         reverse_mode=reverse_mode,
         class_no=class_no,
         save_path=saved_information_path)

    # save model
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    print('\n')
    print('\nTraining finished and model saved\n')

    return model

