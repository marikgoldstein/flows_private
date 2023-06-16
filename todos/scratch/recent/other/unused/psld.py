
Unif = torch.distributions.Uniform
Normal = torch.distributions.Normal


dim = 728
full_dim = 728*2




dxt = 
    beta(t)/2 * (-Gamma xt + M^{-1} mt)  dt
dmt = 
    beta(t)/2 * (-1 xt + -nu mt) dt

+

sqrt(Gamma beta(t)) eps1 root(dt)
sqrt(M nu beta(t)) eps2 root(dt)



lambda = ((Gamma+nu) += sqrt((Gamma-nu)^2 - 4M^{-1}))/2
critical damping M^{-1} = (Gamma-nu)^2 / 4
4 M^{-1} = (Gamma-nu)^2



m0 = N(0, M little gamma I)
Sigma0xm = 0
Sigma0xx = 0
Sigma0mm = M little gamma
m0 = 0
mut = [mutx mutm]
shared = exp[- \frac{nu + Gamma}{4} B(t)]
mutx = (A1 B(t) x0 + A2 B(t) m0 + x0 ) * shared
mutm = (C1 B(t) x0 + C2 B(t) m0 + m0 ) * shared
B(t) = \int_0^t beta(s) ds
A1 = (nu - Gamma) / 4
A2 = (Gamma - nu)^2 / 8
C1 = -1/2 
C2 = (Gamma-nu)/4
#stationary is N(0,I) for x and N(0, MI) for m


Sigma 0 = Sigma0xx, 0 
          0        Sigma0mm

A1 = M^{-1} / 4
A2 = M^{-2} / 8
A3 = (nu - Gamma)/2 
A4 = -M^{-1} / 2
A5 = (Gamma-nu)/2
C1 = (Gamma-nu)/8
C2 = (Gamma-nu)^3/32
C3 = -1/2 
C4 = M^{-1}/2
C5 = (nu-Gamma)/4
D1 = 1/4
D2 = M^{-1}/4
D3 = (Gamma-nu)/2
D4 = -1/2
D5 =  ( M (nu-Gamma))/2







Sigmatxx = A1 * B(t).pow(2) * Sigma0xx
         + A2 * B(t).pow(2) * SIgma0mm
         + A3 * B(t) * Sigma0xx
         + A4 * B(t).pow(2)
         + A5 * B(t)
         + (exp[2 lambda B(t) - 1]
         + Sigma0xx

Sigmatxm = 
        C1 B(t).pow(2) Sigma0xx
    +   C2 B(t).pow(2) * Sigma0mm
    +   C3 B(t) Sigma0xx
    +   C4 B(t) SIgma0mm
    +   C5 B(t).pow(2)

Sigmatmm = 

    D1 B(t).pow(2) Sigma0xx
+   D2 B(t).pow(2) Sigma0mm
+   D3 B(t) Sigma0 mm
+   D4 B(t).pow(2)
+   D5 B(t)
+   M * (exp[2 lambda B(t)] - 1)
+   Sigma0mm











for batch_idx, batch in enumerate(loader):

    x0, _ = batch
    bsz = x.shape[0]
    t = Unif(low=0.0,high=1.0).sample(sample_shape=(bsz,))
    zero = torch.zeros(bsz)
    one = torch.ones(bsz)
    eps = Normal(loc = zero, scale = one).sample(sample_shape=(bsz,full_dim))
    mut = ...
    Lt = ...
    zt = mut + Lt eps 
    eps_hat = eps_theta(zt, t)
    eps_diff = (eps_hat - eps).pow(2)
    assert eps_diff.shape==(bsz, full_dim)
    loss = eps_diff.sum(-1)
    loss = loss.mean(0)



