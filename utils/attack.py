import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinfPGDAttack(object):
    def __init__(self, handler=None, device=None, epsilon=0.05, k=10, a=0.01, rand=True, feat=None):
        """
        FGSM, I-FGSM and PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
        """
        self.handler = handler
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = F.mse_loss
        self.device = device

        # Feature-level attack? Which layer?
        self.feat = feat

        # PGD or I-FGSM?
        self.rand = rand

    def perturb(self, X_nat, y, c_trg):
        """
        X_net is the output of network.
        Vanilla Attack.
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-1e-9, 1e-9, X_nat.shape).astype('float32')).to(self.device)

        grad = 0.0
        for i in range(self.k):
            X.requires_grad = True
            output = self.handler.manipulate(X, c_trg)
            self.handler.model.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad
            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.handler.model.zero_grad()

        return X, eta