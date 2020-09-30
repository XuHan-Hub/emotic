import torch
import torch.nn as nn


class Emotic(nn.Module):
    ''' Emotic Model'''

    # def __init__(self, num_context_features, num_body_features):
    def __init__(self, num_context_features, num_body_features, num_seg_features=0, num_depth_features=0, args=None):
        super(Emotic, self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_seg_features = num_seg_features
        self.num_depth_features = num_depth_features
        self.args = args

        if args.context_depth == True and args.context_seg == True:
            dim = self.num_context_features + self.num_body_features + self.num_seg_features + self.num_depth_features
        elif args.context_depth == True:
            dim = self.num_context_features + self.num_body_features + self.num_depth_features
        elif args.context_seg == True:
            dim = self.num_context_features + self.num_body_features + self.num_seg_features

        self.fc1 = nn.Linear(dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.relu = nn.ReLU()

        # self.fc_seg = nn.Linear(150*28*28,1024)
        # self.fc_att1 = nn.Linear(1000,256)
        # self.fc_att2 = nn.Linear(1000, 256)
        # self.bn_att1 = nn.BatchNorm1d(256)
        # self.bn_att2 = nn.BatchNorm1d(256)

    def forward(self, x_context, x_body, x_context_seg=None, x_context_depth=None):
        # ax, rx = x_context
        # ax  = self.fc_att1(ax)
        # ax = self.bn_att1(ax)
        # ax = self.relu(ax)
        # rx = self.fc_att2(rx)
        # rx = self.bn_att2(rx)
        # rx = self.relu(rx)
        if self.args.context_ABN == True:
            context_features = torch.cat((x_context), 1)
        else:
            context_features = x_context.view(-1, self.num_context_features)

        body_features = x_body.view(-1, self.num_body_features)

        if self.args.context_depth == True and self.args.context_seg == True:
            context_depth_features = x_context_depth.view(-1, self.num_depth_features)
            context_seg_features = x_context_seg.view(-1, self.num_seg_features)
            fuse_features = torch.cat((context_features, body_features, context_seg_features, context_depth_features),
                                      1)
        elif self.args.context_depth == True:
            context_depth_features = x_context_depth.view(-1, self.num_depth_features)
            fuse_features = torch.cat((context_features, body_features, context_depth_features), 1)
        elif self.args.context_seg == True:
            context_seg_features = x_context_seg.view(-1, self.num_seg_features)
            fuse_features = torch.cat((context_features, body_features, context_seg_features), 1)
        else:
            fuse_features = torch.cat((context_features, body_features), 1)

        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        return cat_out, cont_out
