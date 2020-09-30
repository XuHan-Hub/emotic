import os
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torch.nn import functional as F
import torch.nn as nn


def prep_models(context_model='resnet18', body_model='resnet18', model_dir='./'):
    ''' Download imagenet pretrained models for context_model and body_model.
    :param context_model: Model to use for conetxt features.
    :param body_model: Model to use for body features.
    :param model_dir: Directory path where to store pretrained models.
    :return: Yolo model after loading model weights
    '''

    model_name = '%s_places365.pth.tar' % context_model
    model_file = os.path.join(model_dir, model_name)
    if not os.path.exists(model_file):
        download_command = 'wget ' + 'http://places2.csail.mit.edu/models_places365/' + model_name + ' -O ' + model_file
        os.system(download_command)

    save_file = os.path.join(model_dir, '%s_places365_py36.pth.tar' % context_model)
    from functools import partial
    import pickle
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
    torch.save(model, save_file)

    # create the network architecture
    model_context = models.__dict__[context_model](num_classes=365)
    model_context_seg = models.__dict__[context_model](num_classes=365)
    model_context_depth = models.__dict__[context_model](num_classes=365)
    checkpoint = torch.load(save_file, map_location=lambda storage,
                                                           loc: storage)  # model trained in GPU could be deployed in CPU machine like this!
    if context_model == 'densenet161':
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        state_dict = {str.replace(k, 'norm.', 'norm'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'conv.', 'conv'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'normweight', 'norm.weight'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'normrunning', 'norm.running'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'normbias', 'norm.bias'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'convweight', 'conv.weight'): v for k, v in state_dict.items()}
    else:
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
            'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name

    model_context.load_state_dict(state_dict)
    model_context.eval()
    model_context.cpu()
    torch.save(model_context, os.path.join(model_dir, 'context_model' + '.pth'))

    print('completed preparing context model')

    model_body = models.__dict__[body_model](pretrained=True)
    model_body.cpu()
    torch.save(model_body, os.path.join(model_dir, 'body_model' + '.pth'))

    print('completed preparing body model')

    model_context_seg.load_state_dict(state_dict)
    model_context_seg.eval()
    model_context_seg.cpu()
    torch.save(model_context_seg, os.path.join(model_dir, 'context_seg_model' + '.pth'))

    print('completed preparing context seg model')

    model_context_depth.load_state_dict(state_dict)
    model_context_depth.eval()
    model_context_depth.cpu()
    torch.save(model_context_depth, os.path.join(model_dir, 'context_seg_model' + '.pth'))

    print('completed preparing context depth model')

    from ABN.resnet import resnet101, resnet18

    # model = resnet101()
    model_ABN = resnet18(pretrained=True)
    checkpoint = torch.load(model_dir + "/model_best.pth.tar", map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model_ABN.load_state_dict(state_dict)
    print('completed preparing context ABN model')

    return model_context, model_body, model_ABN, model_context_seg, model_context_depth


if __name__ == '__main__':
    prep_models(model_dir='proj/debug_exp/models')
