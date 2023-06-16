import torch
from utils_numerical import (                                                                                                                                     
    cat, stack, zeros, zeros_like, ones, ones_like, randn, randn_like, rand, rand_like,
    flip, sqrt, mat_square_root, matrix_exp, 
    trace_fn,
    batch_transpose,
    eye,
    linspace,
    sqnorm
)   
 

def EM(trainer, n_samples, model, device):

    conf = trainer.config
    diffusion_obj = trainer.diffusion
    assert conf.dgm_type == 'diffusion'
    sde = diffusion_obj.sde
    prior_sample_fn = sde.sample_from_prior
    T_min = conf.T_min
    T_max = conf.T_max
    n_discrete_steps = conf.n_sample_steps - 1
    clip_samples = conf.clip_samples
    tweedie = conf.tweedie

    def reverse_sde(u_t, _t, probability_flow=False):
        # _t has underscore to avoid accidentally using it instead of rev_t 
        # the correct one to use is _rev t 

        batch_size = u_t.shape[0]
        rev_t = 1 - _t
        D = {'u_t':u_t, 't': rev_t}
        rev_s = diffusion_obj.t_to_s(rev_t)
        std = sde.transition_std(rev_t, s=rev_s) # make sure (t, s=0) works okay
        var = std.pow(2)
        mean_coef = sde.transition_mean_coefficient(rev_t, s=rev_s)
        
        D['std'] = std
        D['var'] = var
        D['mean_coef'] = mean_coef
        score_pred_args = {'D': D, 'model': model}
        score_hat = diffusion_obj.score_pred(**score_pred_args)
        f, g, g2 = sde.get_fG(u_t, rev_t)
        g2score = g2[:,None, None, None] * score_hat
        rev_drift = g2score * (0.5 if probability_flow else 1.0) - f
        rev_diff = zeros_like(g) if probability_flow else g
        return rev_drift, rev_diff


    def one_step_EM(t_scalar, dt, u, u_mean):
        n_samples = u.shape[0]                                     
        eps = randn_like(u).type_as(u)                             
        t = t_scalar * ones(n_samples).type_as(u)                  
        drift, diffusion = reverse_sde(u, t)                    
        u_mean = u + drift * dt
        root_dt = torch.sqrt(dt)
        u = u_mean + (diffusion[:,None, None, None] * eps * root_dt)
        return u, u_mean  

    # aggressive
    def clip_func(x):
        return x.clamp(min=-1, max=1)

    def get_sampler_t_array(N, T_begin, T_final):
        t = torch.linspace(T_begin, T_final, N + 1)
        #t = T_final * flip(1 - (t / T_final) ** 2.0, dims=[0]) # quad
        #t = t.clamp(T_begin, T_final)
        return t         
             
    def main_loop(N, ts, u, u_mean, clipping):
        for i in range(N):
            if i % 500 == 0:
                print("sampling, step {} / {}".format(i, N))
            dt = ts[i + 1] - ts[i]
            u, u_mean = one_step_EM(ts[i], dt, u, u_mean)
            if clipping:
                u, u_mean = clip_func(u), clip_func(u_mean)
        return u, u_mean 
             

    def tweedie_step_fn(u_eps, u_eps_mean, T_min, clipping):

        assert T_min <= 1e-3
        assert T_min <= trainer.config.delta, "bad choice for delta < T_min"
        T_min_tensor = torch.tensor([T_min]).type_as(u_eps)

        bsz = u_eps.shape[0]
        eps = T_min_tensor 
        eps_vec = eps * torch.ones(bsz,).type_as(u_eps)
        
        std = sde.transition_std(eps, s=None)
        var = std.pow(2)
        mean_coef = sde.transition_mean_coefficient(eps, s=None)

        # get stheta
        score_pred_args = {
            'D': {'u_t': u_eps, 't': eps_vec, 'std': std, 'var': var, 'mean_coef': mean_coef} ,
            'mode': model,
        }
        score_hat = trainer.score_pred(**score_pred_args)

        #N(x | \frac{x'}{a} + \frac{beta^2}{a}s_\theta(x', eps), variance = \frac{beta^2}{a^2} I). a is mean coef, beta^2 is var
        mu_term1 = u_eps / mean_coef[:, None, None, None]
        mu_term2 = (var / mean_coef)[:, None, None, None] * score_hat
        mu = mu_term1 + mu_term2
        # only need conditional mean
        #variance = var / mean_coef.pow(2)
        #sigma = variance.sqrt()
        return mu



    def denoising_step_fn(u_eps, u_eps_mean, T_min, T_final, clipping):
        # t = t final = .9999 (since it will be reversed)
        # t min is like 1e-5
        #print("Denoising Step: Computing at t = ", T_final, "and dt = ", T_min)
        T_min_tensor = torch.tensor([T_min]).type_as(u_eps)
        u_0, u_0_mean = one_step_EM(T_final, T_min_tensor, u_eps, u_eps_mean)
        if clipping:
            u_0, u_0_mean = clip_func(u_0), clip_func(u_0_mean)
        return u_0, u_0_mean

    print("Sampling")
    u_init = prior_sample_fn(n_samples)
    u_init = u_init.to(device) # should not be needed but just in case
    T_begin = 1.0 - T_max 
    T_final = 1.0 - T_min
    ts = get_sampler_t_array(N=n_discrete_steps, T_begin = T_begin, T_final = T_final)
    u_init = u_init.to(device)
    ts = ts.type_as(u_init)
    with torch.no_grad():
        u_eps, u_eps_mean = main_loop(n_discrete_steps, ts, u_init, u_init, clipping = clip_samples)
       
        if tweedie:
            # dont need t final of .9999, just jump from T_min = 1e-5
            u_0_mean = tweedie_step_fn(u_eps, u_eps_mean, T_min = T_min, clipping = clip_samples) 
        else:
            _, u_0_mean = denoising_step_fn(u_eps, u_eps_mean, T_min = T_min, T_final = T_final, clipping = clip_samples)
    
    # these have not yet been changed from u_0 -> x images
    return u_0_mean

def plot_drift(trainer, real_data, model_samples):

    real_data = real_data.detach().numpy()
    samples = samples.detach().numpy()

    J, K, T = 10, 10, 16
    M = int(np.sqrt(T))
    MIN, MAX = -30, 30
    # make spatial inputs
    x0 = linspace(MIN, MAX, steps=J)
    x1 = linspace(MIN, MAX, steps=K)
    inputs = []
    for j in range(J):
        for k in range(K):
            inputs.append(torch.tensor([x0[j], x1[k]]))
    inputs = torch.stack(inputs,dim=0)
    N = J*K 
    assert inputs.shape==(N, 2)

    # define times and plot
    ts = linspace(0, 1.0, T)
    f, axes = plt.subplots(M,M, figsize=(12,12))
    # compute fields
    for i, t in enumerate(ts):
        axi, axj = int(i/M), i % M
        ax = axes[axi,axj]
        ax.grid(False)
        assert inputs.shape==(N,2)
        with torch.no_grad():
            tarr = ones(N) * t
            fields_i = trainer.reverse_sde(inputs, tarr, probability_flow=False)[0]
            fields_i = fields_i.detach().numpy()
        for j in range(N):
            inx,iny = inputs[j,0], inputs[j,1]
            outx,outy = fields_i[j,0], fields_i[j,1]
            ax.quiver(inx,iny,outx,outy)
        ax.set_title("time {}".format(round(t.item(),3))) 
        ax.scatter(real_data[:,0], real_data[:,1],label='real')
        ax.scatter(samples[:,0], samples[:,1],label='model samples')
        ax.set_xlim(MIN, MAX)
        ax.set_ylim(MIN, MAX)
    plt.xlim(MIN, MAX)
    plt.ylim(MIN, MAX)
    plt.legend()
    plt.grid()
    plt.show()
