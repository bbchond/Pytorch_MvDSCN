import torch
import torch.nn as nn
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from metric import thrC, post_proC, err_rate, rand_index_score, f1_score
from model import AutoEncoderInit
from torch.backends import cudnn
from tqdm import tqdm


class MsDSCN():
    def __init__(self,
                 views_data,
                 n_samples,
                 label,
                 device=torch.device('cpu'),
                 learning_rate=1e-3,
                 weight_decay=0.00,
                 epochs=600,
                 ft=False,
                 random_seed=41,
                 alpha=1,
                 beta=0.1,
                 theta=0.1,
                 lamda=0.1,
                 loss_type='mse',
                 model_path=None,
                 show_res=10):
        self.views_data = views_data
        self.learning_rate = learning_rate
        self.device = device
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.ft = ft
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda = lamda
        self.loss_type = loss_type
        self.batch_size = n_samples
        self.model_path = model_path
        self.show_res = show_res
        self.label = label

        fix_seed(random_seed)

    def HSIC(self, c_v, c_w):
        N = c_v.shape[0]
        H = torch.ones((N, N)) * ((1 / N) * (-1)) + torch.eye(N)
        H = H.cuda()
        K_1 = torch.matmul(c_v, c_v.t()).cuda()
        K_2 = torch.matmul(c_w, c_w.t()).cuda()
        rst = torch.matmul(K_1, H).cuda()
        rst = torch.matmul(rst, K_2).cuda()
        rst = torch.matmul(rst, H).cuda()
        rst = torch.trace(rst).cuda()
        return rst

    def train(self):
        views_data = self.views_data
        views_data[0] = views_data[0].to(self.device)
        views_data[1] = views_data[1].to(self.device)

        model = AutoEncoderInit(batch_size=self.batch_size, ft=self.ft)
        model = model.to(self.device)

        if self.ft:  # load parameters from the init_pretrained_autoencoder
            print("============loading pretrained params============")
            pre_trained_model = torch.load(self.model_path)
            parameters_initAE = dict([(name, param) for name, param in pre_trained_model.named_parameters()])
            for name, param in model.named_parameters():
                if name in parameters_initAE:
                    param_pre = parameters_initAE[name]
                    param.data = param_pre.data
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_values = []
        epoch_iter = tqdm(range(self.epochs))
        for epoch in epoch_iter:
            model.train()
            if self.ft:
                view1_out, view2_out, view1_r_out, view2_r_out, z1, z2, z_common, diversity_latent_1, \
                diversity_latent_2, latent1, latent2, latent1_diversity_se, latent2_diversity_se, latent1_se, \
                latent2_se = model(views_data)
                view1_rec_loss = torch.sum(torch.pow(view1_out - views_data[0], 2.0)) + \
                                 torch.sum(torch.pow(view1_r_out - views_data[0], 2.0))
                view2_rec_loss = torch.sum(torch.pow(view2_out - views_data[1], 2.0)) + \
                                 torch.sum(torch.pow(view2_r_out - views_data[1], 2.0))
                view_1_se_loss = torch.sum(torch.pow(latent1 - latent1_se, 2.0)) + \
                                 torch.sum(torch.pow(diversity_latent_1 - latent1_diversity_se, 2.0))
                view_2_se_loss = torch.sum(torch.pow(latent2 - latent2_se, 2.0)) + \
                                 torch.sum(torch.pow(diversity_latent_2 - latent2_diversity_se, 2.0))
                # loss of reconstruction
                reconstruct_loss = view1_rec_loss + view2_rec_loss
                # loss of self-expression
                expression_loss = view_1_se_loss + view_2_se_loss
                # cof regularization
                reg_loss = torch.sum(torch.pow(z1, 2.0)) + torch.sum(torch.pow(z2, 2.0)) + \
                           torch.sum(torch.pow(z_common, 2.0))
                # unify loss
                unify_loss = torch.sum(torch.abs(z_common - z1)) + torch.sum(torch.abs(z_common - z2))
                hsic_loss = self.HSIC(z1, z2)
                loss = reconstruct_loss + self.alpha * expression_loss + self.beta * reg_loss + 0.1 * unify_loss + 0.1 * hsic_loss
                if (epoch + 1) % self.show_res == 0:
                    alpha = max(0.4 - (self.label.shape[0] - 1) / 10 * 0.1, 0.1)
                    Coef = thrC(z_common.detach().cpu().numpy(), alpha)
                    y_hat, L = post_proC(Coef, self.label.max(), 3, 1)
                    missrate_x = err_rate(self.label, y_hat)
                    acc_x = 1 - missrate_x
                    nmi = normalized_mutual_info_score(self.label, y_hat)
                    f_measure = f1_score(self.label, y_hat)
                    ri = rand_index_score(self.label, y_hat)
                    ar = adjusted_rand_score(self.label, y_hat)
                    print("nmi: %.4f" % nmi, "accuracy: %.4f" % acc_x, "F-measure: %.4f" % f_measure, "RI: %.4f" % ri,
                          "AR: %.4f" % ar)

            else:
                view1_out, view2_out, view1_r_out, view2_r_out = model(views_data)
                # loss = F.mse_loss(data, out, reduction="none")
                # loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
                loss1 = 0.5 * torch.sum(torch.pow(torch.sub(view1_out, views_data[0]), 2.0))
                loss2 = 0.5 * torch.sum(torch.pow(torch.sub(view2_out, views_data[1]), 2.0))
                loss3 = 0.5 * torch.sum(torch.pow(torch.sub(view1_r_out, views_data[0]), 2.0))
                loss4 = 0.5 * torch.sum(torch.pow(torch.sub(view2_r_out, views_data[1]), 2.0))
                loss = loss1 + loss2 + loss3 + loss4
                # loss = loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.ft:
                epoch_iter.set_description(f"# Epoch {epoch}, train_loss: {loss.item():.4f}, "
                                           f"rec_loss: {reconstruct_loss.item():.4f}, "
                                           f"self_exp_loss: {expression_loss.item():.4f}, "
                                           f"reg_loss: {reg_loss.item():.4f}, "
                                           f"hisc-loss: {hsic_loss.item():.4f}")
            else:
                epoch_iter.set_description(
                    f"# Epoch {epoch}, train_loss: {loss.item():.4f}, loss1: {loss1.item():.4f}, loss2: {loss2.item():.4f}, loss3: {loss3.item():.4f},loss4: {loss4.item():.4f}")

            loss_values.append(loss.item())
        plt.plot(np.linspace(1, self.epochs, self.epochs).astype(int), loss_values)
        return model


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
