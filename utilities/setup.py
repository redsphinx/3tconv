from torchvision.models import resnet18, googlenet, vgg19_bn, vgg16_bn
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
import torch
import os

from config import paths as PP
from models.resnet18 import ResNet18Explicit, ResNet18Explicit3DConv
from models.googlenet import Googlenet3TConv_explicit, Googlenet3DConv_explicit


def get_model(project_variable):
    # project_variable = ProjectVariable()
    if type(project_variable.load_model) == bool:
        print('loading weights from torchvision model')
    elif project_variable.load_model is not None:
        if len(project_variable.load_model) == 3:
            ex, mo, ep = project_variable.load_model
            path = os.path.join(PP.models, 'experiment_%d_model_%d' % (ex, mo), 'epoch_%d' % ep)
        else:
            ex, mo, ep, run = project_variable.load_model
            path = os.path.join(PP.models, 'experiment_%d_model_%d_run_%d' % (ex, mo, run), 'epoch_%d' % ep)

        if not os.path.exists(path):
            print("ERROR: saved model path '%s' does not exist" % path)
            return None
    else:
        path, ex, mo, ep = None, None, None, None

    if project_variable.model_number == 20:
        # model = ResNet18(project_variable)
        model = ResNet18Explicit(project_variable)
        if type(project_variable.load_model) != bool and not project_variable.load_model is None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        elif project_variable.load_model:
            # load resnet18 from pytorch
            tmp_resnet18 = resnet18(pretrained=True)
            # copy the weights
            divide = False
            if divide:
                div1 = 7.
                div2 = 3.
            else:
                div1 = 1.
                div2 = 1.
            model.conv1.first_weight = torch.nn.Parameter(tmp_resnet18.conv1.weight.unsqueeze(2) / div1)
            model.conv2.first_weight = torch.nn.Parameter(tmp_resnet18.layer1[0].conv1.weight.unsqueeze(2) / div2)
            model.conv3.first_weight = torch.nn.Parameter(tmp_resnet18.layer1[0].conv2.weight.unsqueeze(2) / div2)
            model.conv4.first_weight = torch.nn.Parameter(tmp_resnet18.layer1[1].conv1.weight.unsqueeze(2) / div2)
            model.conv5.first_weight = torch.nn.Parameter(tmp_resnet18.layer1[1].conv2.weight.unsqueeze(2) / div2)
            model.conv6.weight = torch.nn.Parameter(tmp_resnet18.layer2[0].downsample[0].weight.unsqueeze(2) / div2)
            model.conv7.first_weight = torch.nn.Parameter(tmp_resnet18.layer2[0].conv1.weight.unsqueeze(2) / div2)
            model.conv8.first_weight = torch.nn.Parameter(tmp_resnet18.layer2[0].conv2.weight.unsqueeze(2) / div2)
            model.conv9.first_weight = torch.nn.Parameter(tmp_resnet18.layer2[1].conv1.weight.unsqueeze(2) / div2)
            model.conv10.first_weight = torch.nn.Parameter(tmp_resnet18.layer2[1].conv2.weight.unsqueeze(2) / div2)
            model.conv11.weight = torch.nn.Parameter(tmp_resnet18.layer3[0].downsample[0].weight.unsqueeze(2) / div2)
            model.conv12.first_weight = torch.nn.Parameter(tmp_resnet18.layer3[0].conv1.weight.unsqueeze(2) / div2)
            model.conv13.first_weight = torch.nn.Parameter(tmp_resnet18.layer3[0].conv2.weight.unsqueeze(2) / div2)
            model.conv14.first_weight = torch.nn.Parameter(tmp_resnet18.layer3[1].conv1.weight.unsqueeze(2) / div2)
            model.conv15.first_weight = torch.nn.Parameter(tmp_resnet18.layer3[1].conv2.weight.unsqueeze(2) / div2)
            model.conv16.weight = torch.nn.Parameter(tmp_resnet18.layer4[0].downsample[0].weight.unsqueeze(2) / div2)
            model.conv17.first_weight = torch.nn.Parameter(tmp_resnet18.layer4[0].conv1.weight.unsqueeze(2) / div2)
            model.conv18.first_weight = torch.nn.Parameter(tmp_resnet18.layer4[0].conv2.weight.unsqueeze(2) / div2)
            model.conv19.first_weight = torch.nn.Parameter(tmp_resnet18.layer4[1].conv1.weight.unsqueeze(2) / div2)
            model.conv20.first_weight = torch.nn.Parameter(tmp_resnet18.layer4[1].conv2.weight.unsqueeze(2) / div2)

        # set weights of 3D conv to not require grad
        model.conv1.weight.requires_grad = False
        model.conv2.weight.requires_grad = False
        model.conv3.weight.requires_grad = False
        model.conv4.weight.requires_grad = False
        model.conv5.weight.requires_grad = False
        model.conv7.weight.requires_grad = False
        model.conv8.weight.requires_grad = False
        model.conv9.weight.requires_grad = False
        model.conv10.weight.requires_grad = False
        model.conv12.weight.requires_grad = False
        model.conv13.weight.requires_grad = False
        model.conv14.weight.requires_grad = False
        model.conv15.weight.requires_grad = False
        model.conv17.weight.requires_grad = False
        model.conv18.weight.requires_grad = False
        model.conv19.weight.requires_grad = False
        model.conv20.weight.requires_grad = False

    elif project_variable.model_number == 21:
        model = ResNet18Explicit3DConv(project_variable)
        if type(project_variable.load_model) != bool and not project_variable.load_model is None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        elif project_variable.load_model:
            # load resnet18 from pytorch
            tmp_resnet18 = resnet18(pretrained=True)
            # copy the weights
            divide = True
            if divide:
                div1 = 7.
                div2 = 3.
            else:
                div1 = 1.
                div2 = 1.

            model.conv1.weight = torch.nn.Parameter(tmp_resnet18.conv1.weight.unsqueeze(2).repeat_interleave(7, dim=2) / div1)
            model.conv2.weight = torch.nn.Parameter(tmp_resnet18.layer1[0].conv1.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv3.weight = torch.nn.Parameter(tmp_resnet18.layer1[0].conv2.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv4.weight = torch.nn.Parameter(tmp_resnet18.layer1[1].conv1.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv5.weight = torch.nn.Parameter(tmp_resnet18.layer1[1].conv2.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv6.weight = torch.nn.Parameter(tmp_resnet18.layer2[0].downsample[0].weight.unsqueeze(2))
            model.conv7.weight = torch.nn.Parameter(tmp_resnet18.layer2[0].conv1.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv8.weight = torch.nn.Parameter(tmp_resnet18.layer2[0].conv2.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv9.weight = torch.nn.Parameter(tmp_resnet18.layer2[1].conv1.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv10.weight = torch.nn.Parameter(tmp_resnet18.layer2[1].conv2.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv11.weight = torch.nn.Parameter(tmp_resnet18.layer3[0].downsample[0].weight.unsqueeze(2))
            model.conv12.weight = torch.nn.Parameter(tmp_resnet18.layer3[0].conv1.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv13.weight = torch.nn.Parameter(tmp_resnet18.layer3[0].conv2.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv14.weight = torch.nn.Parameter(tmp_resnet18.layer3[1].conv1.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv15.weight = torch.nn.Parameter(tmp_resnet18.layer3[1].conv2.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv16.weight = torch.nn.Parameter(tmp_resnet18.layer4[0].downsample[0].weight.unsqueeze(2))
            model.conv17.weight = torch.nn.Parameter(tmp_resnet18.layer4[0].conv1.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv18.weight = torch.nn.Parameter(tmp_resnet18.layer4[0].conv2.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv19.weight = torch.nn.Parameter(tmp_resnet18.layer4[1].conv1.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv20.weight = torch.nn.Parameter(tmp_resnet18.layer4[1].conv2.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)

    elif project_variable.model_number == 23:
        model = Googlenet3TConv_explicit(project_variable)
        if type(project_variable.load_model) != bool and not project_variable.load_model is None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        elif project_variable.load_model:
            # load googlenet from pytorch
            tmp_googlenet = googlenet(pretrained=True, transform_input=False)

            model.conv1.first_weight = torch.nn.Parameter(tmp_googlenet.conv1.conv.weight.unsqueeze(2))
            model.conv2.weight = torch.nn.Parameter(tmp_googlenet.conv2.conv.weight.unsqueeze(2))
            model.conv3.first_weight = torch.nn.Parameter(tmp_googlenet.conv3.conv.weight.unsqueeze(2))
            # inception 3a
            model.conv4.weight = torch.nn.Parameter(tmp_googlenet.inception3a.branch1.conv.weight.unsqueeze(2))
            model.conv5.weight = torch.nn.Parameter(tmp_googlenet.inception3a.branch2[0].conv.weight.unsqueeze(2))
            model.conv6.first_weight = torch.nn.Parameter(tmp_googlenet.inception3a.branch2[1].conv.weight.unsqueeze(2))
            model.conv7.weight = torch.nn.Parameter(tmp_googlenet.inception3a.branch3[0].conv.weight.unsqueeze(2))
            model.conv8.first_weight = torch.nn.Parameter(tmp_googlenet.inception3a.branch3[1].conv.weight.unsqueeze(2))
            model.conv9.weight = torch.nn.Parameter(tmp_googlenet.inception3a.branch4[1].conv.weight.unsqueeze(2))
            # inception 3b
            model.conv10.weight = torch.nn.Parameter(tmp_googlenet.inception3b.branch1.conv.weight.unsqueeze(2))
            model.conv11.weight = torch.nn.Parameter(tmp_googlenet.inception3b.branch2[0].conv.weight.unsqueeze(2))
            model.conv12.first_weight = torch.nn.Parameter(tmp_googlenet.inception3b.branch2[1].conv.weight.unsqueeze(2))
            model.conv13.weight = torch.nn.Parameter(tmp_googlenet.inception3b.branch3[0].conv.weight.unsqueeze(2))
            model.conv14.first_weight = torch.nn.Parameter(tmp_googlenet.inception3b.branch3[1].conv.weight.unsqueeze(2))
            model.conv15.weight = torch.nn.Parameter(tmp_googlenet.inception3b.branch4[1].conv.weight.unsqueeze(2))
            # inception 4a
            model.conv16.weight = torch.nn.Parameter(tmp_googlenet.inception4a.branch1.conv.weight.unsqueeze(2))
            model.conv17.weight = torch.nn.Parameter(tmp_googlenet.inception4a.branch2[0].conv.weight.unsqueeze(2))
            model.conv18.first_weight = torch.nn.Parameter(tmp_googlenet.inception4a.branch2[1].conv.weight.unsqueeze(2))
            model.conv19.weight = torch.nn.Parameter(tmp_googlenet.inception4a.branch3[0].conv.weight.unsqueeze(2))
            model.conv20.first_weight = torch.nn.Parameter(tmp_googlenet.inception4a.branch3[1].conv.weight.unsqueeze(2))
            model.conv21.weight = torch.nn.Parameter(tmp_googlenet.inception4a.branch4[1].conv.weight.unsqueeze(2))
            # inception 4b
            model.conv22.weight = torch.nn.Parameter(tmp_googlenet.inception4b.branch1.conv.weight.unsqueeze(2))
            model.conv23.weight = torch.nn.Parameter(tmp_googlenet.inception4b.branch2[0].conv.weight.unsqueeze(2))
            model.conv24.first_weight = torch.nn.Parameter(tmp_googlenet.inception4b.branch2[1].conv.weight.unsqueeze(2))
            model.conv25.weight = torch.nn.Parameter(tmp_googlenet.inception4b.branch3[0].conv.weight.unsqueeze(2))
            model.conv26.first_weight = torch.nn.Parameter(tmp_googlenet.inception4b.branch3[1].conv.weight.unsqueeze(2))
            model.conv27.weight = torch.nn.Parameter(tmp_googlenet.inception4b.branch4[1].conv.weight.unsqueeze(2))
            # inception 4c
            model.conv29.weight = torch.nn.Parameter(tmp_googlenet.inception4c.branch1.conv.weight.unsqueeze(2))
            model.conv30.weight = torch.nn.Parameter(tmp_googlenet.inception4c.branch2[0].conv.weight.unsqueeze(2))
            model.conv31.first_weight = torch.nn.Parameter(tmp_googlenet.inception4c.branch2[1].conv.weight.unsqueeze(2))
            model.conv32.weight = torch.nn.Parameter(tmp_googlenet.inception4c.branch3[0].conv.weight.unsqueeze(2))
            model.conv33.first_weight = torch.nn.Parameter(tmp_googlenet.inception4c.branch3[1].conv.weight.unsqueeze(2))
            model.conv34.weight = torch.nn.Parameter(tmp_googlenet.inception4c.branch4[1].conv.weight.unsqueeze(2))
            # inception 4d
            model.conv35.weight = torch.nn.Parameter(tmp_googlenet.inception4d.branch1.conv.weight.unsqueeze(2))
            model.conv36.weight = torch.nn.Parameter(tmp_googlenet.inception4d.branch2[0].conv.weight.unsqueeze(2))
            model.conv37.first_weight = torch.nn.Parameter(tmp_googlenet.inception4d.branch2[1].conv.weight.unsqueeze(2))
            model.conv38.weight = torch.nn.Parameter(tmp_googlenet.inception4d.branch3[0].conv.weight.unsqueeze(2))
            model.conv39.first_weight = torch.nn.Parameter(tmp_googlenet.inception4d.branch3[1].conv.weight.unsqueeze(2))
            model.conv40.weight = torch.nn.Parameter(tmp_googlenet.inception4d.branch4[1].conv.weight.unsqueeze(2))
            # inception 4e
            model.conv41.weight = torch.nn.Parameter(tmp_googlenet.inception4e.branch1.conv.weight.unsqueeze(2))
            model.conv42.weight = torch.nn.Parameter(tmp_googlenet.inception4e.branch2[0].conv.weight.unsqueeze(2))
            model.conv43.first_weight = torch.nn.Parameter(tmp_googlenet.inception4e.branch2[1].conv.weight.unsqueeze(2))
            model.conv44.weight = torch.nn.Parameter(tmp_googlenet.inception4e.branch3[0].conv.weight.unsqueeze(2))
            model.conv45.first_weight = torch.nn.Parameter(tmp_googlenet.inception4e.branch3[1].conv.weight.unsqueeze(2))
            model.conv46.weight = torch.nn.Parameter(tmp_googlenet.inception4e.branch4[1].conv.weight.unsqueeze(2))
            # inception 5a
            model.conv48.weight = torch.nn.Parameter(tmp_googlenet.inception5a.branch1.conv.weight.unsqueeze(2))
            model.conv49.weight = torch.nn.Parameter(tmp_googlenet.inception5a.branch2[0].conv.weight.unsqueeze(2))
            model.conv50.first_weight = torch.nn.Parameter(tmp_googlenet.inception5a.branch2[1].conv.weight.unsqueeze(2))
            model.conv51.weight = torch.nn.Parameter(tmp_googlenet.inception5a.branch3[0].conv.weight.unsqueeze(2))
            model.conv52.first_weight = torch.nn.Parameter(tmp_googlenet.inception5a.branch3[1].conv.weight.unsqueeze(2))
            model.conv53.weight = torch.nn.Parameter(tmp_googlenet.inception5a.branch4[1].conv.weight.unsqueeze(2))
            # inception 5b
            model.conv54.weight = torch.nn.Parameter(tmp_googlenet.inception5b.branch1.conv.weight.unsqueeze(2))
            model.conv55.weight = torch.nn.Parameter(tmp_googlenet.inception5b.branch2[0].conv.weight.unsqueeze(2))
            model.conv56.first_weight = torch.nn.Parameter(tmp_googlenet.inception5b.branch2[1].conv.weight.unsqueeze(2))
            model.conv57.weight = torch.nn.Parameter(tmp_googlenet.inception5b.branch3[0].conv.weight.unsqueeze(2))
            model.conv58.first_weight = torch.nn.Parameter(tmp_googlenet.inception5b.branch3[1].conv.weight.unsqueeze(2))
            model.conv59.weight = torch.nn.Parameter(tmp_googlenet.inception5b.branch4[1].conv.weight.unsqueeze(2))


        model.conv1.weight.requires_grad = False
        model.conv3.weight.requires_grad = False
        model.conv6.weight.requires_grad = False
        model.conv8.weight.requires_grad = False
        model.conv12.weight.requires_grad = False
        model.conv14.weight.requires_grad = False
        model.conv18.weight.requires_grad = False
        model.conv20.weight.requires_grad = False
        model.conv24.weight.requires_grad = False
        model.conv26.weight.requires_grad = False
        model.conv31.weight.requires_grad = False
        model.conv33.weight.requires_grad = False
        model.conv37.weight.requires_grad = False
        model.conv39.weight.requires_grad = False
        model.conv43.weight.requires_grad = False
        model.conv45.weight.requires_grad = False
        model.conv50.weight.requires_grad = False
        model.conv52.weight.requires_grad = False
        model.conv56.weight.requires_grad = False
        model.conv58.weight.requires_grad = False


    elif project_variable.model_number == 25:
        model = Googlenet3DConv_explicit(project_variable)
        if type(project_variable.load_model) != bool and not project_variable.load_model is None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        elif project_variable.load_model:
            # load googlenet from pytorch
            tmp_googlenet = googlenet(pretrained=True, transform_input=False)

            div1 = 7.
            div2 = 3.

            model.conv1.weight = torch.nn.Parameter(tmp_googlenet.conv1.conv.weight.unsqueeze(2).repeat_interleave(7, dim=2) / div1)
            model.conv2.weight = torch.nn.Parameter(tmp_googlenet.conv2.conv.weight.unsqueeze(2))
            model.conv3.weight = torch.nn.Parameter(tmp_googlenet.conv3.conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            # inception 3a
            model.conv4.weight = torch.nn.Parameter(tmp_googlenet.inception3a.branch1.conv.weight.unsqueeze(2))
            model.conv5.weight = torch.nn.Parameter(tmp_googlenet.inception3a.branch2[0].conv.weight.unsqueeze(2))
            model.conv6.weight = torch.nn.Parameter(tmp_googlenet.inception3a.branch2[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv7.weight = torch.nn.Parameter(tmp_googlenet.inception3a.branch3[0].conv.weight.unsqueeze(2))
            model.conv8.weight = torch.nn.Parameter(tmp_googlenet.inception3a.branch3[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv9.weight = torch.nn.Parameter(tmp_googlenet.inception3a.branch4[1].conv.weight.unsqueeze(2))
            # inception 3b
            model.conv10.weight = torch.nn.Parameter(tmp_googlenet.inception3b.branch1.conv.weight.unsqueeze(2))
            model.conv11.weight = torch.nn.Parameter(tmp_googlenet.inception3b.branch2[0].conv.weight.unsqueeze(2))
            model.conv12.weight = torch.nn.Parameter(tmp_googlenet.inception3b.branch2[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv13.weight = torch.nn.Parameter(tmp_googlenet.inception3b.branch3[0].conv.weight.unsqueeze(2))
            model.conv14.weight = torch.nn.Parameter(tmp_googlenet.inception3b.branch3[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv15.weight = torch.nn.Parameter(tmp_googlenet.inception3b.branch4[1].conv.weight.unsqueeze(2))
            # inception 4a
            model.conv16.weight = torch.nn.Parameter(tmp_googlenet.inception4a.branch1.conv.weight.unsqueeze(2))
            model.conv17.weight = torch.nn.Parameter(tmp_googlenet.inception4a.branch2[0].conv.weight.unsqueeze(2))
            model.conv18.weight = torch.nn.Parameter(tmp_googlenet.inception4a.branch2[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv19.weight = torch.nn.Parameter(tmp_googlenet.inception4a.branch3[0].conv.weight.unsqueeze(2))
            model.conv20.weight = torch.nn.Parameter(tmp_googlenet.inception4a.branch3[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv21.weight = torch.nn.Parameter(tmp_googlenet.inception4a.branch4[1].conv.weight.unsqueeze(2))
            # inception 4b
            model.conv22.weight = torch.nn.Parameter(tmp_googlenet.inception4b.branch1.conv.weight.unsqueeze(2))
            model.conv23.weight = torch.nn.Parameter(tmp_googlenet.inception4b.branch2[0].conv.weight.unsqueeze(2))
            model.conv24.weight = torch.nn.Parameter(tmp_googlenet.inception4b.branch2[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv25.weight = torch.nn.Parameter(tmp_googlenet.inception4b.branch3[0].conv.weight.unsqueeze(2))
            model.conv26.weight = torch.nn.Parameter(tmp_googlenet.inception4b.branch3[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv27.weight = torch.nn.Parameter(tmp_googlenet.inception4b.branch4[1].conv.weight.unsqueeze(2))
            # inception 4c
            model.conv29.weight = torch.nn.Parameter(tmp_googlenet.inception4c.branch1.conv.weight.unsqueeze(2))
            model.conv30.weight = torch.nn.Parameter(tmp_googlenet.inception4c.branch2[0].conv.weight.unsqueeze(2))
            model.conv31.weight = torch.nn.Parameter(tmp_googlenet.inception4c.branch2[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv32.weight = torch.nn.Parameter(tmp_googlenet.inception4c.branch3[0].conv.weight.unsqueeze(2))
            model.conv33.weight = torch.nn.Parameter(tmp_googlenet.inception4c.branch3[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv34.weight = torch.nn.Parameter(tmp_googlenet.inception4c.branch4[1].conv.weight.unsqueeze(2))
            # inception 4d
            model.conv35.weight = torch.nn.Parameter(tmp_googlenet.inception4d.branch1.conv.weight.unsqueeze(2))
            model.conv36.weight = torch.nn.Parameter(tmp_googlenet.inception4d.branch2[0].conv.weight.unsqueeze(2))
            model.conv37.weight = torch.nn.Parameter(tmp_googlenet.inception4d.branch2[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv38.weight = torch.nn.Parameter(tmp_googlenet.inception4d.branch3[0].conv.weight.unsqueeze(2))
            model.conv39.weight = torch.nn.Parameter(tmp_googlenet.inception4d.branch3[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv40.weight = torch.nn.Parameter(tmp_googlenet.inception4d.branch4[1].conv.weight.unsqueeze(2))
            # inception 4e
            model.conv41.weight = torch.nn.Parameter(tmp_googlenet.inception4e.branch1.conv.weight.unsqueeze(2))
            model.conv42.weight = torch.nn.Parameter(tmp_googlenet.inception4e.branch2[0].conv.weight.unsqueeze(2))
            model.conv43.weight = torch.nn.Parameter(tmp_googlenet.inception4e.branch2[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv44.weight = torch.nn.Parameter(tmp_googlenet.inception4e.branch3[0].conv.weight.unsqueeze(2))
            model.conv45.weight = torch.nn.Parameter(tmp_googlenet.inception4e.branch3[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv46.weight = torch.nn.Parameter(tmp_googlenet.inception4e.branch4[1].conv.weight.unsqueeze(2))
            # inception 5a
            model.conv48.weight = torch.nn.Parameter(tmp_googlenet.inception5a.branch1.conv.weight.unsqueeze(2))
            model.conv49.weight = torch.nn.Parameter(tmp_googlenet.inception5a.branch2[0].conv.weight.unsqueeze(2))
            model.conv50.weight = torch.nn.Parameter(tmp_googlenet.inception5a.branch2[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv51.weight = torch.nn.Parameter(tmp_googlenet.inception5a.branch3[0].conv.weight.unsqueeze(2))
            model.conv52.weight = torch.nn.Parameter(tmp_googlenet.inception5a.branch3[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv53.weight = torch.nn.Parameter(tmp_googlenet.inception5a.branch4[1].conv.weight.unsqueeze(2))
            # inception 5b
            model.conv54.weight = torch.nn.Parameter(tmp_googlenet.inception5b.branch1.conv.weight.unsqueeze(2))
            model.conv55.weight = torch.nn.Parameter(tmp_googlenet.inception5b.branch2[0].conv.weight.unsqueeze(2))
            model.conv56.weight = torch.nn.Parameter(tmp_googlenet.inception5b.branch2[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv57.weight = torch.nn.Parameter(tmp_googlenet.inception5b.branch3[0].conv.weight.unsqueeze(2))
            model.conv58.weight = torch.nn.Parameter(tmp_googlenet.inception5b.branch3[1].conv.weight.unsqueeze(2).repeat_interleave(3, dim=2) / div2)
            model.conv59.weight = torch.nn.Parameter(tmp_googlenet.inception5b.branch4[1].conv.weight.unsqueeze(2))

    if project_variable.model_number == 50:
        model = ResNet18Explicit(project_variable)
        if type(project_variable.load_model) != bool and not project_variable.load_model is None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        elif project_variable.load_model:
            # load resnet18 from pytorch
            tmp_resnet18 = resnet18(pretrained=True)
            # copy the weights
            model.conv1.first_weight = torch.nn.Parameter(tmp_resnet18.conv1.weight.unsqueeze(2))
            model.conv2.first_weight = torch.nn.Parameter(tmp_resnet18.layer1[0].conv1.weight.unsqueeze(2))
            model.conv3.first_weight = torch.nn.Parameter(tmp_resnet18.layer1[0].conv2.weight.unsqueeze(2))
            model.conv4.first_weight = torch.nn.Parameter(tmp_resnet18.layer1[1].conv1.weight.unsqueeze(2))
            model.conv5.first_weight = torch.nn.Parameter(tmp_resnet18.layer1[1].conv2.weight.unsqueeze(2))
            model.conv6.weight = torch.nn.Parameter(tmp_resnet18.layer2[0].downsample[0].weight.unsqueeze(2))
            model.conv7.first_weight = torch.nn.Parameter(tmp_resnet18.layer2[0].conv1.weight.unsqueeze(2))
            model.conv8.first_weight = torch.nn.Parameter(tmp_resnet18.layer2[0].conv2.weight.unsqueeze(2))
            model.conv9.first_weight = torch.nn.Parameter(tmp_resnet18.layer2[1].conv1.weight.unsqueeze(2))
            model.conv10.first_weight = torch.nn.Parameter(tmp_resnet18.layer2[1].conv2.weight.unsqueeze(2))
            model.conv11.weight = torch.nn.Parameter(tmp_resnet18.layer3[0].downsample[0].weight.unsqueeze(2))
            model.conv12.first_weight = torch.nn.Parameter(tmp_resnet18.layer3[0].conv1.weight.unsqueeze(2))
            model.conv13.first_weight = torch.nn.Parameter(tmp_resnet18.layer3[0].conv2.weight.unsqueeze(2))
            model.conv14.first_weight = torch.nn.Parameter(tmp_resnet18.layer3[1].conv1.weight.unsqueeze(2))
            model.conv15.first_weight = torch.nn.Parameter(tmp_resnet18.layer3[1].conv2.weight.unsqueeze(2))
            model.conv16.weight = torch.nn.Parameter(tmp_resnet18.layer4[0].downsample[0].weight.unsqueeze(2))
            model.conv17.first_weight = torch.nn.Parameter(tmp_resnet18.layer4[0].conv1.weight.unsqueeze(2))
            model.conv18.first_weight = torch.nn.Parameter(tmp_resnet18.layer4[0].conv2.weight.unsqueeze(2))
            model.conv19.first_weight = torch.nn.Parameter(tmp_resnet18.layer4[1].conv1.weight.unsqueeze(2))
            model.conv20.first_weight = torch.nn.Parameter(tmp_resnet18.layer4[1].conv2.weight.unsqueeze(2))

        # set weights of 3D conv to not require grad
        # TODO: check gradients for srxy -> should be False,, and the mlp -> should be True
        model.conv1.weight.requires_grad = False
        model.conv2.weight.requires_grad = False
        model.conv3.weight.requires_grad = False
        model.conv4.weight.requires_grad = False
        model.conv5.weight.requires_grad = False
        model.conv7.weight.requires_grad = False
        model.conv8.weight.requires_grad = False
        model.conv9.weight.requires_grad = False
        model.conv10.weight.requires_grad = False
        model.conv12.weight.requires_grad = False
        model.conv13.weight.requires_grad = False
        model.conv14.weight.requires_grad = False
        model.conv15.weight.requires_grad = False
        model.conv17.weight.requires_grad = False
        model.conv18.weight.requires_grad = False
        model.conv19.weight.requires_grad = False
        model.conv20.weight.requires_grad = False
    
    else:
        print('ERROR: model_number=%d not supported' % project_variable.model_number)
        model = None

    return model


def get_optimizer(project_variable, model):
    # project_variable = ProjectVariable()

    if project_variable.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=project_variable.learning_rate)

    elif project_variable.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=project_variable.learning_rate, momentum=project_variable.momentum)

    else:
        print('Error: optimizer %s not supported' % project_variable.optimizer)
        optimizer = None

    return optimizer


def get_device(project_variable):
    if project_variable.device is None:
        _dev = 'cpu'
    elif type(project_variable.device) is int:
        _dev = 'cuda:%d' % project_variable.device
    else:
        _dev = None

    device = torch.device(_dev)

    return device