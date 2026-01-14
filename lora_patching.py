import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import denorm, Image2tensor
from torchvision.utils import save_image, make_grid
from net.lora4conv import inject_lora
from deepfake import DeepfakeHandler
from utils.attack import LinfPGDAttack
from utils.BLIP_loss import BLIPDistanceCalculator
from torchvision.models import resnet50
torch.set_printoptions(precision=3)


class LoRA_patching:

    def __init__(self, device, args):

        self.device = device
        self.model_type = args.deepfake
        self.lr = 0.001
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.epochs = args.epochs
        self.warning = args.warning
        self.rank = args.rank
        self.leakage = args.leakage
        self.batch_size = args.batch_size
        self.lambda_feat = args.lambda_feat
        self.lambda_blip = args.lambda_blip
        self.save_interval = 20

        self.warning_image = Image2tensor("data/warning.png")

        # The original deepfake
        self.deepfake_ori_handler = DeepfakeHandler(device, self.model_type)
        for param in self.deepfake_ori_handler.model.parameters():
            param.requires_grad = False
        
        # deepfake with lora patch
        self.deepfake_handler = DeepfakeHandler(device, self.model_type)
        # patching the deepfake model
        inject_lora(module=self.deepfake_handler.model, rank=self.rank, alpha=2.0, gated=True, gated_type="basic", freeze_norm=True)
        self.deepfake_handler.model = self.deepfake_handler.model.to(device)
        # The parameters of the convolutional and deconvolutional layers have been frozen
        self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.deepfake_handler.model.parameters()),
                                            self.lr, [self.beta1, self.beta2])
        if args.mode == "train":
            self.print_param_counts()
        
        # Loss Components:
        # self.blip_loss = BLIPDistanceCalculator(device=device)

        resnet = resnet50(pretrained=True).eval().to(device)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.save_path = self.get_save_path()
    
    def print_param_counts(self):
        total_params = sum(p.numel() for p in self.deepfake_handler.model.parameters())
        frozen_params = sum(p.numel() for p in self.deepfake_handler.model.parameters() if not p.requires_grad)
        trainable_params = total_params - frozen_params
        print(f"#Total parameters: {total_params:,}")
        print(f" -Trainable parameters: {trainable_params:,}")
        print(f" -Frozen parameters: {frozen_params:,}")



    def get_save_path(self, root="checkpoint/lora/lora_patch4"):
        save_name = root + self.model_type
        print("Patched model:", self.model_type)
        if self.warning:
            save_name += "_warning"
            print("loading warning mechanism")
        print(f"rank = {self.rank}")
        return save_name + ".pth"


    def feature_loss(self, img1, img2):
        features1 = self.feature_extractor(img1)
        features2 = self.feature_extractor(img2)
        return F.mse_loss(features1, features2)


    def loss_fn(self, y_output_ori, y_output, y_output_adv):
        if self.warning:
            #ground truth of the warning watermark image
            y_output_ori = torch.clamp(self.warning_image * 2 + y_output_ori, -1, 1)
        
        # Pixel-level loss
        L_diff = F.mse_loss(y_output, y_output_ori) + F.mse_loss(y_output_adv, y_output_ori)
        # Image feature loss
        # L_feat = (self.feature_loss(y_output, y_output_ori) +
        #             self.feature_loss(y_output_adv, y_output_ori)) * self.lambda_feat
        L_feat = torch.tensor([0]).to(self.device)
        # Semantic feature loss
        # L_blip = (self.blip_loss(y_output, y_output_ori) + self.blip_loss(y_output_adv, y_output_ori)) * self.lambda_blip
        L_blip = torch.tensor([0]).to(self.device)
    
        return L_diff, L_feat, L_blip


    def train(self, train_dataloader):
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.
            total_diff_loss = 0.
            total_feat_loss = 0.
            total_blip_loss = 0.

            with tqdm(total=len(train_dataloader),desc=f'Epoch {epoch}/{self.epochs}',unit='batch') as pbar:
                
                for n, (x_input, c_org) in enumerate(train_dataloader):
                    x_input, c_org = x_input.to(self.device), c_org.to(self.device)
                    c_org_list = self.deepfake_handler.process_input(c_org)

                    for c_trg in c_org_list:
                        with torch.no_grad():
                            # y_output_now is used to solve the problem of fighting Audi
                            y_output_now = self.deepfake_handler.manipulate(x_input, c_trg)
                            # y_output_ori is used to calculate the loss
                            y_output_ori = self.deepfake_ori_handler.manipulate(x_input, c_trg)

                        # Adversarial Training Paradigm
                        attack = LinfPGDAttack(handler=self.deepfake_handler, device=self.device, epsilon=0.05)
                        x_adv, _ = attack.perturb(x_input, y_output_now, c_trg)
                        
                        y_output = self.deepfake_handler.manipulate(x_input, c_trg)
                        y_output_adv = self.deepfake_handler.manipulate(x_adv, c_trg)

                        diff_loss, feat_loss, blip_loss = self.loss_fn(y_output_ori, y_output, y_output_adv)

                        loss = diff_loss + feat_loss + blip_loss

                        self.optimizer_G.zero_grad()
                        loss.backward()
                        self.optimizer_G.step()

                        with torch.no_grad():
                            total_loss += loss / len(c_org_list)
                            total_diff_loss += diff_loss / len(c_org_list)
                            total_feat_loss += feat_loss / len(c_org_list)
                            total_blip_loss += blip_loss / len(c_org_list)

                    if n % self.save_interval == 0:
                        save_res = torch.cat([x_input, y_output_ori, y_output,y_output_adv], dim=0)
                        save_image(denorm(save_res), f"save/train_res/{self.model_type}_{n}.jpg", nrow=self.batch_size)
                        # torch.save(self.deepfake.state_dict(), self.save_path)

                    pbar.set_postfix(
                        total_loss=total_loss.item() / (n + 1),
                        diff_loss=total_diff_loss.item() / (n + 1),
                        feat_loss=total_feat_loss.item() / (n + 1),
                        blip_loss=total_blip_loss.item() / (n + 1))
                    pbar.update()
                    
                torch.save(self.deepfake_handler.model.state_dict(), self.save_path)

    def test(self, test_dataloader):
        self.deepfake_handler.model.load_state_dict(torch.load(self.save_path, 
                                                               map_location=lambda storage, loc: storage))
        self.deepfake_handler.model = self.deepfake_handler.model.to(self.device)

        for n, (x_input, c_org) in enumerate(tqdm(test_dataloader)):
            x_input, c_org = x_input.to(self.device), c_org.to(self.device)
            c_org_list = self.deepfake_handler.process_input(c_org)
            y_ori_list = []
            y_adv_ori_list = []
            y_lora_list = []
            y_adv_lora_list = []

            for _, c_trg in enumerate(c_org_list):
                if not self.leakage:
                    x_adv = self.deepfake_ori_handler.proactive_defend(x_input, c_trg)
                    x_adv_ori = x_adv.clone()
                else:
                    x_adv = self.deepfake_handler.proactive_defend(x_input, c_trg)
                    x_adv_ori = self.deepfake_ori_handler.proactive_defend(x_input, c_trg)
                
                with torch.no_grad():   
                    y_ori = self.deepfake_ori_handler.manipulate(x_input, c_trg)
                    y_ori_list.append(denorm(y_ori))

                    y_adv_ori = self.deepfake_ori_handler.manipulate(x_adv_ori, c_trg)
                    y_adv_ori_list.append(denorm(y_adv_ori))

                    y_lora = self.deepfake_handler.manipulate(x_input, c_trg)
                    y_lora_list.append(denorm(y_lora))

                    y_adv_lora = self.deepfake_handler.manipulate(x_adv, c_trg)
                    y_adv_lora_list.append(denorm(y_adv_lora))

            lists = [y_ori_list, y_adv_ori_list, y_lora_list, y_adv_lora_list]

            rows = [torch.cat(img_list, dim=0) for img_list in lists]
            full_batch = torch.cat(rows, dim=0)
            grid = make_grid(full_batch, nrow=len(c_org_list), padding=2)
            save_image(grid, f"save/test_res/{self.model_type}_{n}.jpg")
