import torch
from PIL import Image
from torchvision import transforms as T
from utils.data_loader import get_loader
import yaml
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    if x.min() < 0:
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    else:
        return x

def resize_image(self, image, sizes):
    return F.interpolate(image,
                            size=(sizes, sizes),
                            mode='bilinear',
                            align_corners=True)

# process celebA labels
def create_labels(c_org, c_dim=5, selected_attrs=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
            hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        c_trg = c_org.clone()
        if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
            c_trg[:, i] = 1
            for j in hair_color_indices:
                if j != i:
                    c_trg[:, j] = 0
        else:
            c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
        c_trg_list.append(c_trg.to(device))
    return c_trg_list


def check_attribute_conflict(att_batch, att_name, att_names):
    def _get(att, att_name):
        if att_name in att_names:
            return att[att_names.index(att_name)]
        return None
    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value
    att_id = att_names.index(att_name)
    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] != 0:
            if _get(att, 'Bangs') != 0:
                _set(att, 1-att[att_id], 'Bangs')
        elif att_name == 'Bangs' and att[att_id] != 0:
            for n in ['Bald', 'Receding_Hairline']:
                if _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] != 0:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] != 0:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Mustache', 'No_Beard'] and att[att_id] != 0:
            for n in ['Mustache', 'No_Beard']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
    return att_batch


def imFromAttReg(att, reg, x_real):
    """Mixes attention, color and real images"""
    return (1-att)*reg + att*x_real


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def Image2tensor(imagepath, resize=256):
    img = Image.open(imagepath).convert("RGB")
    transform = []
    transform.append(T.ToTensor())
    if len(img.split()) == 3:
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    else:
        transform.append(T.Normalize(mean=0.5, std=0.5))
    transform.append(T.Resize([resize,resize]))

    transform = T.Compose(transform)
    img = torch.unsqueeze(transform(img), dim=0).to(device)
    return img


def getDataloader(image_dir, attr_path, image_size=256, batch_size=4, data_size=1000, model_type="stargan", ):
    celeba_image_dir = image_dir
    attr_path = attr_path

    if model_type != "attgan":
        selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    else:
        selected_attrs = ["Bald","Bangs","Black_Hair","Blond_Hair","Brown_Hair","Bushy_Eyebrows","Eyeglasses","Male",
                      "Mouth_Slightly_Open","Mustache","No_Beard","Pale_Skin","Young"]

    train_loader = get_loader(celeba_image_dir, attr_path, selected_attrs, image_size=image_size, batch_size=batch_size,
                              data_size=data_size, shuffle=False,mode="train", num_workers=0)
    test_loader = get_loader(celeba_image_dir, attr_path, selected_attrs, image_size=image_size, batch_size=1,
                             data_size=data_size, mode="test", num_workers=0)

    return train_loader,test_loader