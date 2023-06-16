import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from sklearn import linear_model as lm
from sklearn.linear_model import LinearRegression

def to_np(x):
    return x.cpu().detach().numpy()

class ControlRegression:

    def __init__(self,):

        self.reg = None

    def fit(self, x, y):
        #reg = lm.Lasso(alpha=2.0) 
        self.reg = lm.LinearRegression()
        self.reg.fit(to_np(x), to_np(y))

    def predict(self, x):
        return torch.tensor(self.reg.predict(to_np(x))).type_as(x)

    def fit_predict(self, x_train, y_train, inputs, inputs_mean):
        self.fit(x_train, y_train)
        g = self.predict(inputs)
        g_mu = self.predict(inputs_mean)
        return g - g_mu

    def check_r_squared(self, x, y):
        preds = self.predict(x)
        squared_residuals = (y - preds).pow(2)
        squared_dev = (y - y.mean()).pow(2)
        r_sq = 1 - (squared_residuals.sum() / squared_dev.sum())
        return r_sq
                                                                                               

class ControlModule:

    def __init__(self, datamodule):

        self.datamodule = datamodule


    def control_scalar(self, x_train, x_val, x_train_mean, x_val_mean, grad_train, grad_val):
        g_diff_val = ControlRegression().fit_predict(x_train, grad_train, x_val, x_val_mean)
        control_val = grad_val - g_diff_val
        g_diff_train = ControlRegression().fit_predict(x_val, grad_val, x_train, x_train_mean)
        control_train = grad_train - g_diff_train
        control = torch.cat([control_train, control_val], dim=0)
        return control


    def control_grads(self, batch, grads):

        feats_for_control, split = self.datamodule.make_reg_features(batch)
        x_train, x_val, x_train_mean, x_val_mean = feats_for_control

        out = {}

        for param_name in grads:

            grad = grads[param_name]
            new_grad = torch.zeros_like(grad)

            if len(grad.shape) == 3:

                for i in range(grad.shape[1]):
                    for j in range(grad.shape[2]):
                        grad_train, grad_val = grad[:split, i, j], grad[split:, i, j]
                        new_grad[:,i,j] = self.control_scalar(
                                x_train,
                                x_val,
                                x_train_mean, 
                                x_val_mean, 
                                grad_train, 
                                grad_val
                        )

            elif len(grad.shape) == 2:

                for i in range(grad.shape[1]):
                    grad_train, grad_val = grad[:split, i], grad[split:, i]
                    new_grad[:,i] = self.control_scalar(
                            x_train, 
                            x_val, 
                            x_train_mean, 
                            x_val_mean, 
                            grad_train, 
                            grad_val
                    )
            else:

                assert False

            out[param_name] = new_grad

        return out
