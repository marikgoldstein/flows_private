import torch
import numpy as np

# noise to score, f, omega

class Converter:

    def __init__(self,):
        self.hi = 'hi'

    def noise_to_score(self, D):
        return -D['noise'] / D['std'][:, None]

    def noise_to_f(self, D):
        score = self.noise_to_score(D)
        D['score'] = score
        f = self.score_to_f(D)
        return f

    def noise_to_omega(self, D):
        return -D['noise'] * D['std'][:,None]

    # score to noise ,f, omega

    def score_to_noise(self, D):
        return -D['score'] * D['std'][:, None]

    def score_to_omega(self, D):
        return D['score'] * D['var'][:, None]

    def score_to_f(self, D):
        return (D['score'] * D['var'][:, None] + D['u_t']) / D['mean_coef'][:, None]

    # omega to score, noise ,f 

    def omega_to_noise(self, D):
        return -D['omega'] / D['std'][:,None]

    def omega_to_score(self, D):
        return D['omega'] / D['var'][:, None]

    def omega_to_f(self, D):
        score = self.omega_to_score(D)
        D['score'] = score
        f = self.score_to_f(D)
        return f

    # f to score, noise, omega

    def f_to_score(self, D):
        return (D['f'] * D['mean_coef'][:, None] - D['u_t']) / D['var'][:, None]

    def f_to_omega(self, D):
        score = self.f_to_score(D)
        D['score'] = score
        omega = self.score_to_omega(D)
        return omega

    def f_to_noise(self, D):
        score = self.f_to_score(D)
        D['score'] = score
        noise = self.score_to_noise(D)
        return noise

    # trainer.C, .H, .W
    def call_model_on_flat(self, model_obj, D, with_ema_model):

        parameterization = model_obj.model_parameterization
        u_t = D['u_t']
        t = D['t']
        bsz = u_t.shape[0]
        C, H, W = model_obj.trainer.C, model_obj.trainer.H, model_obj.trainer.W
        u_t = u_t.reshape(bsz, C, H, W)
        out = model_obj(u_t, t, with_ema_model)
        out = out.reshape(bsz, C * H * W)
        return out

    def convert(self, param, desired, D):

        if param == desired:
            return D[param]
        else:
            fn = getattr(self, f'{param}_to_{desired}')
            return fn(D)

    def noise_pred(self, model_obj, D, with_ema_model):
        p = model_obj.model_parameterization
        D[p] = self.call_model_on_flat(model_obj, D, with_ema_model)
        return self.convert(p, 'noise', D)

    def omega_pred(self, model_obj, D, with_ema_model):
        p = model_obj.model_parameterization
        D[p] = self.call_model_on_flat(model_obj, D, with_ema_model)
        return self.convert(p, 'omega', D)

    def score_pred(self, model_obj, D, with_ema_model):
        p = model_obj.model_parameterization
        D[p] = self.call_model_on_flat(model_obj, D, with_ema_model)
        return self.convert(p, 'score', D)

    def f_pred(self, model_obj, D, with_ema_model):
        p = model_obj.model_parameterization
        D[p] = self.call_model_on_flat(model_obj, D, with_ema_model)
        return self.convert(p, 'f', D)
