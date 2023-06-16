import torch
import torch.nn as nn
import sys
import matplotlib as mpl
from matplotlib import pyplot as plt
from functorch import vmap
import seaborn as sns
import numpy as np
from math import pi
import time
import torch.distributions as D

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from utils import Config, GMM
from models import TinyNet

#######################################
######### DEFINING THE FLOWS ##########
#######################################

def phi(x0, x1, t):
    _t = t[:,None]
    xt = _t*x1 + (1-_t)*x0
    return xt

def dphi(x0, x1, t):
    dt = x1 - x0
    return dt

# assumes xt = tx1 + (1-t)x0
def conditional_density_path(xt, t, x1):
    assert xt.shape==x1.shape
    tx1 = t.unsqueeze(-1) * x1
    onemt = (1 - t).unsqueeze(-1).repeat(1,2)
    inverted = (xt-tx1)/onemt
    recon = tx1 + onemt*inverted # to test inversion
    deriv = 1 / (1-t)
    log_prior = q0.log_prob(inverted) 
    log_det = 2 * deriv.log()
    log_prob = log_prior + log_det
    prob = log_prob.exp()
    return prob

# assumes xt = tx1 + (1-t)x0
def conditional_field(x, t, x1):
    assert x.shape==x1.shape
    num = (x1-x)
    assert num.shape==x.shape
    denom = (1-t).unsqueeze(-1)
    result = num / denom
    return result

# (consistent) monte carlo estimate of marginal 
# field using the marginalization equation in
# flow matching paper:
# ut(x) = 1/pt(x) * E_{q(x1)}\Bigg[ ut(x|x1)pt(x|x1) \Bigg]
# uses K samples of x1 for each x being evaluated
def estimate_marginal_field(x, t):
    bsz = x.shape[0]
    assert x.shape == (bsz, 2)
    assert t.shape == (bsz,)
    K = 1000 # num samples of x1 per estimate of pt(xt)
    x1s = q1.sample(sample_shape=(bsz,K))
    assert x1s.shape==(bsz,K,2)
    x, t = x.unsqueeze(1), t.unsqueeze(1)
    assert x.shape==(bsz,1,2)
    assert t.shape==(bsz,1)
    x, t = x.repeat(1,K,1), t.repeat(1,K)
    assert x.shape==(bsz,K,2)
    assert t.shape==(bsz,K)
    x = x.reshape(bsz*K,2)
    x1s = x1s.reshape(bsz*K,2)
    t = t.reshape(bsz*K,)
    cond_prob = conditional_density_path(x, t, x1s)
    cond_field = conditional_field(x, t, x1s)
    assert cond_prob.shape==(bsz*K,)
    assert cond_field.shape==(bsz*K,2)
    numer = cond_prob.unsqueeze(-1) * cond_field
    assert numer.shape==(bsz*K,2)
    numer = numer.reshape(bsz, K, 2)
    denom = cond_prob.reshape(bsz, K)
    numer = numer.mean(1)
    denom = denom.mean(1)
    assert numer.shape == (bsz, 2)
    assert denom.shape == (bsz,)
    out = numer / denom.unsqueeze(-1)
    return out



#######################################
######### END FLOW STUFF THE FLOWS ####
#######################################

##############################################
##### HELPERS FOR TRAINING AND PLOTTING ######
##############################################


def get_trajectory(N=256):

    x0 = q0.sample(sample_shape=(N,))
    x0 += torch.ones_like(x0)
    xt = x0
    traj = []
    for i in range(1000):
        dt = 1 / 1000
        t = i*dt
        t = torch.ones(N,) * t
        xt = xt + model(xt, t)*dt
        traj.append(xt)
    traj = torch.stack(traj,dim=0)
    assert traj.shape==(1000, N, 2)
    coord0 = traj.detach().numpy()[:,:,0]
    coord1 = traj.detach().numpy()[:,:,1]
    return coord0,coord1

def plot_trajectories(ax,x,y):
    LEN, N = x.shape
    #fig = plt.figure()
    for i in range(N):
        xi,yi = x[:,i],y[:,i]
        t = range(LEN)
        ax.plot(xi, yi)


# written in a particular way taking step as an arg to
# support matplotlib animation
def training(step):

    for rep in range(opt_steps_per_sample):
        opt.zero_grad()
        x0 = q0.sample(sample_shape=(bsz,))
        x1 = q1.sample(sample_shape=(bsz,))
        t = torch.rand(bsz)
        xt = phi(x0, x1, t)
        ut = dphi(x0, x1, t)
        v = model(xt, t)
        loss = (v - ut).pow(2).sum(-1)
        assert loss.shape==(bsz,)
        loss = loss.mean()
        loss.backward()
        print("loss:{}".format(loss.item()))
        opt.step()

    with torch.no_grad():
        ax1.clear()
        N = 512
        x0 = q0.sample(sample_shape=(N,))
        x1 = q1.sample(sample_shape=(N,))
        ax1.scatter(x0.numpy()[:,0], x0.numpy()[:,1], label='x0')
        ax1.scatter(x1.numpy()[:,0], x1.numpy()[:,1], label='x1')
        xt = x0
        avg_traj = []
        for i in range(1000):
            dt = 1 / 1000
            t = i*dt
            t = torch.ones(N,) * t
            xt = xt + model(xt, t)*dt
        ax1.scatter(xt.detach().numpy()[:,0], xt.detach().numpy()[:,1],label='samples')
        plt.legend()
        plt.title("step:{}".format(step))

##############################################
##### END HELPERS FOR TRAINING AND PLOTTING ##
##############################################


if __name__=='__main__':


    # init stuff
    gmm = GMM(Config(layout='diffusion', N0=1, N1=10))
    q0, q1, d = gmm.q0, gmm.q1, gmm.d
    model = TinyNet(d=d)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bsz = 256
    opt_steps_per_sample = 50
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])


    # main training. "opt_steps_per_sample" number of optimization steps per frame
    ani = animation.FuncAnimation(fig, training, repeat=False, frames=100)
    plt.show()

    xb,yb = get_trajectory(N=256)
    xa,ya = get_trajectory(N=256)
    ax = plt.subplot(1,1,1)
    plot_trajectories(ax,xb,yb)
    plot_trajectories(ax,xa,ya)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xlim(-20,20)
    plt.ylim(-20, 20)
    plt.show()

    # estimate the true marginal vectore field implied by the path
    print("estimating marginal vector field")
    STEPS = 16
    N = 16
    MIN= -25
    MAX= 25
    M = int(np.sqrt(STEPS))
    ts = torch.linspace(0, .90, STEPS).numpy()
    x = np.linspace(MIN,MAX,N)
    y = np.linspace(MIN,MAX,N)
    f, axes = plt.subplots(M,M, figsize=(12,12))
    with torch.no_grad():
        for i,t in enumerate(ts):  
            print("t",t)
            u, v = np.zeros((N,N)), np.zeros((N, N))
            axi, axj = int(i/M), i % M
            ax = axes[axi,axj]
            ax.grid(False)
            for j in range(N):
                for k in range(N):
                    #xval, yval = xmesh[j,k], ymesh[j,k]
                    xval, yval = x[j], y[k]
                    xy = torch.tensor([xval,yval]).unsqueeze(0)
                    tvec = torch.ones(1)*t
                    out = estimate_marginal_field(xy, tvec).numpy()
                    u_tjk, v_tjk = out[0,0], out[0,1]
                    u[j,k] = u_tjk
                    v[j,k] = v_tjk
                    #print(xval,yval,u_tjk,v_tjk)
                    ax.quiver(xval, yval, u_tjk, v_tjk)
            ax.set_title("time {}".format(round(t.item(),3)))
        plt.xlim(MIN, MAX)
        plt.ylim(MIN, MAX)
        plt.grid()
        plt.show()

    # plot model vector field
    print("plotting model vector field")
    f, axes = plt.subplots(M,M, figsize=(12,12))
    with torch.no_grad():
        for i,t in enumerate(ts):  
            u, v = np.zeros((N,N)), np.zeros((N, N))
            axi, axj = int(i/M), i % M
            ax = axes[axi,axj]
            for j in range(N):
                for k in range(N):
                    #xval, yval = xmesh[j,k], ymesh[j,k]
                    xval, yval = x[j], y[k]
                    xy = torch.tensor([xval,yval]).unsqueeze(0)
                    tvec = torch.ones(1)*t
                    xy = xy.float()
                    tvec = tvec.float()
                    out = model(xy, tvec).numpy()
                    u_tjk, v_tjk = out[0,0], out[0,1]
                    u[j,k] = u_tjk
                    v[j,k] = v_tjk
                    ax.quiver(xval, yval, u_tjk, v_tjk)
            ax.set_title("time {}".format(round(t.item(),3)))
        plt.xlim(MIN, MAX)
        plt.ylim(MIN, MAX)
        plt.grid()
        plt.show()

