import torch
import model.stargan as stargan
from model.HiSD.trainer import HiSD_Trainer
from model.attgan import AttGAN
from utils.utils import create_labels, check_attribute_conflict, get_config
from PIL import Image
from torchvision import transforms
from utils.attack import LinfPGDAttack

class DeepfakeHandler:
    def __init__(self, device, model_type="stargan"):
        """
        model_type: str, one of ["stargan", "attgan", "HiSD"]
        """
        self.model_type = model_type
        self.device = device
        self.model = self._load_model(model_type)
        print(f"==== Model loaded successfully: {self.model_type} =====")

    # ----------------------------
    # Model Loading
    # ----------------------------
    def _load_model(self, model_type):
        if model_type == "stargan":
            model = stargan.Generator(conv_dim=64, c_dim=5, repeat_num=6)
            G_path = "checkpoint/deepfake/stargan/200000-G.ckpt"
            model.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            return model.to(self.device)

        elif model_type == "attgan":
            attgan = AttGAN(imagesize=256)
            attgan.load("checkpoint/deepfake/attgan/weights.199.pth")
            return attgan.G.to(self.device)

        elif model_type == "HiSD":
            config = get_config('model/HiSD/configs/celeba-hq_256.yaml')
            checkpoint = 'checkpoint/deepfake/HiSD/gen_00600000.pt'
            trainer = HiSD_Trainer(config)
            state_dict = torch.load(checkpoint)
            trainer.models.gen.load_state_dict(state_dict['gen_test'])
            return trainer.models.gen.to(self.device)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    # ----------------------------
    # Pre-processing
    # ----------------------------
    def process_input(self, c_org=None):
        if self.model_type == "stargan":
            selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
            return create_labels(c_org, 5, selected_attrs)

        elif self.model_type == "attgan":
            test_int = 1.0
            thres_int = 0.5
            selected_attrs = [
                "Bald","Bangs","Black_Hair","Blond_Hair","Brown_Hair",
                "Bushy_Eyebrows","Eyeglasses","Male","Mouth_Slightly_Open",
                "Mustache","No_Beard","Pale_Skin","Young"
            ]
            c_org = c_org.to(self.device).float()
            att_list, c_trg_list = [], []
            for i in range(len(selected_attrs)):
                if i not in [1, 2, 3, 4, 7]:
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

        elif self.model_type == "HiSD":
            config = get_config('model/HiSD/configs/celeba-hq_256.yaml')
            image_size = config['new_size']
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            references = []
            for i in range(3):
                ref_path = f'model/HiSD/examples/reference_haircolor_{i}.jpg'
                ref_img = transform(Image.open(ref_path).convert('RGB')).unsqueeze(0).to(self.device)
                references.append([2, ref_img])
            return references

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    # ----------------------------
    # Manipulation
    # ----------------------------
    def manipulate(self, img, c_ref):
        if self.model_type == "stargan":
            return self.model(img, c_ref)

        elif self.model_type == "attgan":
            self.model.eval()
            try:
                return self.model.G(img, c_ref)
            except:
                return self.model(img, c_ref)

        elif self.model_type == "HiSD":
            self.model.eval()
            type_num, r = c_ref
            s_trg = self.model.extract(r, type_num)
            c_trg = self.model.translate(self.model.encode(img), s_trg, type_num)
            return self.model.decode(c_trg)

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    # ----------------------------
    # Defense with Adversarial Attack
    # ----------------------------
    def proactive_defend(self, img, c_ref, method="Disrupting"):
        if method == "Disrupting":
            attack = LinfPGDAttack(handler=self, device=self.device, epsilon=0.05)
            with torch.no_grad():
                output = self.manipulate(img, c_ref)
            x_adv, _ = attack.perturb(img, output, c_ref)
        else:
            raise ValueError(f"Unsupported defend method: {method}")
        return x_adv
