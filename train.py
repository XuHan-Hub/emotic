import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
from tensorboardX import SummaryWriter

from emotic import Emotic
from emotic_dataset import Emotic_PreDataset
from loss import DiscreteLoss, ContinuousLoss_SL1, ContinuousLoss_L2
from prepare_models import prep_models
from test import test_data


def train_data(opt, scheduler, models, device, train_loader, val_loader, disc_loss, cont_loss, train_writer, val_writer,
               model_path, args):
    '''
    Training emotic model on train data using train loader.
    :param opt: Optimizer object.
    :param scheduler: Learning rate scheduler object.
    :param models: List containing model_context, model_body and emotic_model (fusion model) in that order. 
    :param device: Torch device. Used to send tensors to GPU if available. 
    :param train_loader: Dataloader iterating over train dataset. 
    :param val_loader: Dataloader iterating over validation dataset. 
    :param disc_loss: Discrete loss criterion. Loss measure between discrete emotion categories predictions and the target emotion categories. 
    :param cont_loss: Continuous loss criterion. Loss measure between continuous VAD emotion predictions and the target VAD values.
    :param train_writer: SummaryWriter object to save train logs. 
    :param val_writer: SummaryWriter object to save validation logs. 
    :param model_path: Directory path to save the models after training. 
    :param args: Runtime arguments.
    '''

    model_context, model_body, emotic_model, model_context_seg, model_context_depth = models

    emotic_model.to(device)
    model_context.to(device)
    model_body.to(device)

    if args.context_depth == True and args.context_seg == True:
        model_context_seg.to(device)
        model_context_depth.to(device)
    elif args.context_depth == True:
        model_context_depth.to(device)
    elif args.context_seg == True:
        model_context_seg.to(device)

    print('starting training')

    for e in range(args.epochs):

        running_loss = 0.0
        running_cat_loss = 0.0
        running_cont_loss = 0.0

        emotic_model.train()
        model_context.train()
        model_body.train()

        if args.context_depth == True and args.context_seg == True:
            model_context_seg.train()
            model_context_depth.train()
        elif args.context_depth == True:
            model_context_depth.train()
        elif args.context_seg == True:
            model_context_seg.train()

        # train models for one epoch
        for images_context, images_body, images_context_mask, images_context_seg, images_context_depth, labels_cat, labels_cont in iter(
                train_loader):

            if args.context_mask == True:
                images_context = images_context_mask.to(device)
            else:
                images_context = images_context.to(device)
            images_body = images_body.to(device)

            labels_cat = labels_cat.to(device)
            labels_cont = labels_cont.to(device)

            if args.context_depth == True and args.context_seg == True:
                images_context_seg = images_context_seg.to(device)
                images_context_depth = images_context_depth.to(device)
            elif args.context_depth == True:
                images_context_depth = images_context_depth.to(device)
            elif args.context_seg == True:
                images_context_seg = images_context_seg.to(device)

            opt.zero_grad()

            pred_context = model_context(images_context)

            pred_body = model_body(images_body)

            if args.context_depth == True and args.context_seg == True:
                pred_context_seg = model_context_seg(images_context_seg)
                pred_context_depth = model_context_depth(images_context_depth)
                pred_cat, pred_cont = emotic_model(pred_context, pred_body, pred_context_seg=pred_context_seg,
                                                   pred_context_depth=pred_context_depth)
            elif args.context_depth == True:
                pred_context_depth = model_context_depth(images_context_depth)
                pred_cat, pred_cont = emotic_model(pred_context, pred_body, pred_context_depth=pred_context_depth)
            elif args.context_seg == True:
                pred_context_seg = model_context_seg(images_context_seg)
                pred_cat, pred_cont = emotic_model(pred_context, pred_body, pred_context_seg=pred_context_seg)
            else:
                pred_cat, pred_cont = emotic_model(pred_context, pred_body)

            cat_loss_batch = disc_loss(pred_cat, labels_cat)
            cont_loss_batch = cont_loss(pred_cont * 10, labels_cont * 10)

            loss = (args.cat_loss_weight * cat_loss_batch) + (args.cont_loss_weight * cont_loss_batch)

            running_loss += loss.item()
            running_cat_loss += cat_loss_batch.item()
            running_cont_loss += cont_loss_batch.item()

            loss.backward()
            opt.step()

        if e % 1 == 0:
            print('epoch = %d loss = %.4f cat loss = %.4f cont_loss = %.4f' % (
                e, running_loss, running_cat_loss, running_cont_loss))

        train_writer.add_scalar('losses/total_loss', running_loss, e)
        train_writer.add_scalar('losses/categorical_loss', running_cat_loss, e)
        train_writer.add_scalar('losses/continuous_loss', running_cont_loss, e)

        running_loss = 0.0
        running_cat_loss = 0.0
        running_cont_loss = 0.0

        emotic_model.eval()
        model_context.eval()
        model_body.eval()

        if args.context_depth == True and args.context_seg == True:
            model_context_seg.eval()
            model_context_depth.eval()
        elif args.context_depth == True:
            model_context_depth.eval()
        elif args.context_seg == True:
            model_context_seg.eval()

        with torch.no_grad():
            # validation for one epoch

            for images_context, images_body, images_context_mask, images_context_seg, images_context_depth, labels_cat, labels_cont in iter(
                    val_loader):
                if args.context_mask == True:
                    images_context = images_context_mask.to(device)
                else:
                    images_context = images_context.to(device)
                images_body = images_body.to(device)
                if args.context_depth == True and args.context_seg == True:
                    images_context_seg = images_context_seg.to(device)
                    images_context_depth = images_context_depth.to(device)
                elif args.context_depth == True:
                    images_context_depth = images_context_depth.to(device)
                elif args.context_seg == True:
                    images_context_seg = images_context_seg.to(device)

                labels_cat = labels_cat.to(device)
                labels_cont = labels_cont.to(device)

                pred_context = model_context(images_context)
                pred_body = model_body(images_body)

                if args.context_depth == True and args.context_seg == True:
                    pred_context_seg = model_context_seg(images_context_seg)
                    pred_context_depth = model_context_depth(images_context_depth)
                    pred_cat, pred_cont = emotic_model(pred_context, pred_body, pred_context_seg=pred_context_seg,
                                                       pred_context_depth=pred_context_depth)
                elif args.context_depth == True:
                    pred_context_depth = model_context_depth(images_context_depth)
                    pred_cat, pred_cont = emotic_model(pred_context, pred_body, pred_context_depth=pred_context_depth)
                elif args.context_seg == True:
                    pred_context_seg = model_context_seg(images_context_seg)
                    pred_cat, pred_cont = emotic_model(pred_context, pred_body, pred_context_seg=pred_context_seg)
                else:
                    pred_cat, pred_cont = emotic_model(pred_context, pred_body)

                cat_loss_batch = disc_loss(pred_cat, labels_cat)
                cont_loss_batch = cont_loss(pred_cont * 10, labels_cont * 10)
                loss = (args.cat_loss_weight * cat_loss_batch) + (args.cont_loss_weight * cont_loss_batch)

                running_loss += loss.item()
                running_cat_loss += cat_loss_batch.item()
                running_cont_loss += cont_loss_batch.item()

        if e % 1 == 0:
            print('epoch = %d validation loss = %.4f cat loss = %.4f cont loss = %.4f ' % (
                e, running_loss, running_cat_loss, running_cont_loss))

        val_writer.add_scalar('losses/total_loss', running_loss, e)
        val_writer.add_scalar('losses/categorical_loss', running_cat_loss, e)
        val_writer.add_scalar('losses/continuous_loss', running_cont_loss, e)

        scheduler.step()

    print('completed training')
    emotic_model.to("cpu")
    model_context.to("cpu")
    model_body.to("cpu")
    torch.save(emotic_model, os.path.join(model_path, 'model_emotic_xu.pth'))
    torch.save(model_context, os.path.join(model_path, 'model_context_xu.pth'))
    torch.save(model_body, os.path.join(model_path, 'model_body_xu.pth'))

    if args.context_depth == True and args.context_seg == True:
        torch.save(model_context_depth, os.path.join(model_path, 'model_context_depth_xu.pth'))
        torch.save(model_context_seg, os.path.join(model_path, 'model_context_seg_xu.pth'))
    elif args.context_depth == True:
        torch.save(model_context_depth, os.path.join(model_path, 'model_context_depth_xu.pth'))
    elif args.context_seg == True:
        torch.save(model_context_seg, os.path.join(model_path, 'model_context_seg_xu.pth'))

    print('saved models')


def train_emotic(result_path, model_path, train_log_path, val_log_path, ind2cat, ind2vad, context_norm, body_norm,
                 context_mask_norm, context_seg_norm, context_depth_norm,
                 args):
    ''' Prepare dataset, dataloders, models. 
    :param result_path: Directory path to save the results (val_predidictions mat object, val_thresholds npy object).
    :param model_path: Directory path to load pretrained base models and save the models after training. 
    :param train_log_path: Directory path to save the training logs. 
    :param val_log_path: Directoty path to save the validation logs. 
    :param ind2cat: Dictionary converting integer index to categorical emotion. 
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
    :param context_norm: List containing mean and std values for context images. 
    :param body_norm: List containing mean and std values for body images. 
    :param args: Runtime arguments. 
    '''
    # Load preprocessed data from npy files
    train_context = np.load(os.path.join(args.data_path, 'train_context_arr.npy'))
    train_context_mask = np.load(os.path.join(args.data_path, 'train_context_mask_arr.npy'))
    train_context_depth = np.load(os.path.join(args.data_path, 'train_context_depth_arr.npy'))
    train_context_seg = np.load(os.path.join(args.data_path, 'train_context_seg_arr.npy'))
    train_body = np.load(os.path.join(args.data_path, 'train_body_arr.npy'))
    train_cat = np.load(os.path.join(args.data_path, 'train_cat_arr.npy'))
    train_cont = np.load(os.path.join(args.data_path, 'train_cont_arr.npy'))

    val_context = np.load(os.path.join(args.data_path, 'val_context_arr.npy'))
    val_context_mask = np.load(os.path.join(args.data_path, 'val_context_mask_arr.npy'))
    val_context_depth = np.load(os.path.join(args.data_path, 'val_context_mask_arr.npy'))
    val_context_seg = np.load(os.path.join(args.data_path, 'val_context_mask_arr.npy'))
    val_body = np.load(os.path.join(args.data_path, 'val_body_arr.npy'))
    val_cat = np.load(os.path.join(args.data_path, 'val_cat_arr.npy'))
    val_cont = np.load(os.path.join(args.data_path, 'val_cont_arr.npy'))

    print('train ', 'context ', train_context.shape, 'body', train_body.shape, 'cat ', train_cat.shape, 'cont',
          train_cont.shape)
    print('val ', 'context ', val_context.shape, 'body', val_body.shape, 'cat ', val_cat.shape, 'cont', val_cont.shape)

    # Initialize Dataset and DataLoader 
    train_transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                          transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    train_dataset = Emotic_PreDataset(train_context, train_body, train_context_mask, train_context_seg,
                                      train_context_depth, train_cat, train_cont, train_transform, context_norm,
                                      body_norm, context_mask_norm, context_seg_norm, context_depth_norm)
    val_dataset = Emotic_PreDataset(val_context, val_body, val_context_mask, val_context_seg, val_context_depth,
                                    val_cat, val_cont, test_transform, context_norm, body_norm, context_mask_norm,
                                    context_seg_norm, context_depth_norm)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)

    print('train loader ', len(train_loader), 'val loader ', len(val_loader))

    # Prepare models 
    model_context, model_body, model_ABN, model_context_seg, model_context_depth = prep_models(
        context_model=args.context_model, body_model=args.body_model, model_dir=model_path)

    emotic_model = Emotic(list(model_context.children())[-1].in_feature, list(model_body.children())[-1].in_features,
                          512, 512, args)
    if args.context_ABN == True:
        model_context = model_ABN
    else:
        model_context = nn.Sequential(*(list(model_context.children())[:-1]))
    model_body = nn.Sequential(*(list(model_body.children())[:-1]))
    model_context_seg = nn.Sequential(*(list(model_context_seg.children())[:-1]))
    model_context_depth = nn.Sequential(*(list(model_context_depth.children())[:-1]))

    for param in emotic_model.parameters():
        param.requires_grad = True
    for param in model_context.parameters():
        param.requires_grad = True
    for param in model_body.parameters():
        param.requires_grad = True

    device = torch.device("cuda:%s" % (str(args.gpu)) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    if args.context_depth == True and args.context_seg == True:
        for param in model_context_seg.parameters():
            param.requires_grad = True
        for param in model_context_depth.parameters():
            param.requires_grad = True
        opt = optim.Adam((list(emotic_model.parameters()) + list(model_context.parameters()) + list(
            model_body.parameters()) + list(model_context_depth.parameters()) + list(model_context_seg.parameters())),
                         lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.context_depth == True:
        for param in model_context_depth.parameters():
            param.requires_grad = True
        opt = optim.Adam((list(emotic_model.parameters()) + list(model_context.parameters()) + list(
            model_body.parameters()) + list(model_context_depth.parameters())),
                         lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.context_seg == True:
        for param in model_context_seg.parameters():
            param.requires_grad = True
        opt = optim.Adam((list(emotic_model.parameters()) + list(model_context.parameters()) + list(
            model_body.parameters()) + list(model_context_seg.parameters())),
                         lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        opt = optim.Adam((list(emotic_model.parameters()) + list(model_context.parameters()) + list(
            model_body.parameters())),
                         lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(opt, step_size=7, gamma=0.1)
    disc_loss = DiscreteLoss(args.discrete_loss_weight_type, device)
    if args.continuous_loss_type == 'Smooth L1':
        cont_loss = ContinuousLoss_SL1()
    else:
        cont_loss = ContinuousLoss_L2()

    train_writer = SummaryWriter(train_log_path)
    val_writer = SummaryWriter(val_log_path)

    # training
    train_data(opt, scheduler, [model_context, model_body, emotic_model, model_context_seg, model_context_depth],
               device, train_loader, val_loader, disc_loss, cont_loss, train_writer, val_writer, model_path, args)
    # validation
    test_data([model_context, model_body, emotic_model, model_context_seg, model_context_depth], device, val_loader,
              ind2cat, ind2vad, len(val_dataset), result_dir=result_path, test_type='val', args=args)
