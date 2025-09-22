import torch
import os
import torch.nn as nn
from torch import Tensor
import model.stargan as stargan
from model.HiSD.trainer import HiSD_Trainer
from model.attgan import AttGAN
import numpy as np
from utils.utils import create_labels, check_attribute_conflict,get_config
from PIL import Image
from torchvision import transforms
from torchvision import transforms as T
from utils.attack import LinfPGDAttack

device = 'cuda' if torch.cuda.is_available() else 'cpu'


"""
---------------------------------------------------------------
                            stargan
---------------------------------------------------------------                      
"""

def stargan_model(conv_dim=64, c_dim=5, repeat_num=6):
    starG = stargan.Generator(conv_dim=conv_dim, c_dim=c_dim, repeat_num=repeat_num)
    G_path = "checkpoint/deepfake/stargan/200000-G.ckpt"
    starG.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    starG = starG.to(device)
    return starG


def processorg_stargan(c_org,c_dim=5):
    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    c_trg_list = create_labels(c_org, c_dim, selected_attrs)
    return c_trg_list


def stargan_fake(img, c_trg, starG):
    gen_img = starG(img, c_trg)
    return gen_img


"""
---------------------------------------------------------------
                            attgan
---------------------------------------------------------------                      
"""

def attgan_model(imagesize=256):
    attgan = AttGAN(imagesize)
    attgan.load("checkpoint/deepfake/attgan/weights.199.pth")
    return attgan.G

def processorg_attgan(c_org):
    test_int = 1.0
    thres_int = 0.5
    selected_attrs = ["Bald","Bangs","Black_Hair","Blond_Hair","Brown_Hair","Bushy_Eyebrows","Eyeglasses","Male","Mouth_Slightly_Open","Mustache","No_Beard","Pale_Skin","Young"]
    c_org = c_org.to(device)
    c_org = c_org.type(torch.float)
    att_list = []
    c_trg_list = []
    for i in range(len(selected_attrs)):
        if i not in [1,2,3,4,7]:
            continue
        tmp = c_org.clone()
        tmp[:, i] = 1 - tmp[:, i]
        tmp = check_attribute_conflict(tmp, selected_attrs[i], selected_attrs)
        att_list.append(tmp)
    for i, att_b in enumerate(att_list):
        att_b_ = (att_b * 2 - 1) * thres_int
        att_b_[..., i - 1] = att_b_[..., i - 1] * test_int / thres_int
        c_trg_list.append(att_b_)
    return c_trg_list

def attgan_fake(img, att_b_, attG):
    attG.eval()
    try:
        gen_img = attG.G(img, att_b_)
    except:
        gen_img = attG(img, att_b_)
    return gen_img


"""
---------------------------------------------------------------
                            HiSD
---------------------------------------------------------------                      
"""
def HiSD_model():
    config = get_config('model/HiSD/configs/celeba-hq_256.yaml')
    checkpoint = 'checkpoint/deepfake/HiSD/gen_00600000.pt'
    trainer = HiSD_Trainer(config)
    state_dict = torch.load(checkpoint)
    trainer.models.gen.load_state_dict(state_dict['gen_test'])
    trainer.models.gen.to(device)
    return trainer.models.gen


def processref_HiSD(c_org=None):
    config = get_config('model/HiSD/configs/celeba-hq_256.yaml')
    noise_dim = config['noise_dim']
    image_size = config['new_size']
    reference = []
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    reference0 = 'model/HiSD/examples/reference_haircolor_0.jpg'
    reference1 = 'model/HiSD/examples/reference_haircolor_1.jpg'
    reference2 = 'model/HiSD/examples/reference_haircolor_2.jpg'
    reference.append([2,transform(Image.open(reference0).convert('RGB')).unsqueeze(0).to(device)])
    reference.append([2,transform(Image.open(reference1).convert('RGB')).unsqueeze(0).to(device)])
    reference.append([2,transform(Image.open(reference2).convert('RGB')).unsqueeze(0).to(device)])
    return reference


def HiSD_fake(img, reference, model):
    model.eval()
    type_num = reference[0]
    r = reference[1]
    s_trg = model.extract(r, type_num)
    c_trg = model.translate(model.encode(img), s_trg, type_num)
    gen = model.decode(c_trg)
    return gen


"""
---------------------------------------------------------------
                            Summary
---------------------------------------------------------------                      
"""
def load_model(model_type="stargan"):
    if model_type == "stargan":
        model = stargan_model()
    elif model_type == "attgan":
        model = attgan_model()
    elif model_type == "HiSD":
        model = HiSD_model()
    else:
        raise Exception('Unsupported deepfake model:' + model_type)
    return model


def load_process(model_type="stargan"):
    if model_type == "stargan":
        process = processorg_stargan
    elif model_type == "attgan":
        process = processorg_attgan
    elif model_type == "HiSD":
        process = processref_HiSD
    else:
        raise Exception('Unsupported deepfake model:' + model_type)
    return process


def manipulate(img, c_ref, model_type="stargan", model=None):
    if model_type == "stargan":
        result = stargan_fake(img, c_ref, model)
    elif model_type == "attgan":
        result = attgan_fake(img, c_ref, model)
    elif model_type == "HiSD":
        result = HiSD_fake(img, c_ref, model)
    else:
        raise Exception('Unsupported deepfake model:' + model_type)
    return result



def deepfake_defense(img, c_ref, model, model_type="stargan"):
    attack = LinfPGDAttack(model=model,
                            device=device,
                            epsilon=0.05,
                            manipulate=manipulate,
                            com=True)
    with torch.no_grad():
        output = manipulate(img, c_ref, model_type, model)
    x_adv, _ = attack.perturb(img, output, c_ref, model_type)

    return x_adv

