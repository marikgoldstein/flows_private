import torch
import numpy as np

s = 0.008 # some offset for numerical safety
bsz = 1

def cos_term(t):
    numer = t+s
    denom = 1+s
    coef = np.pi / 2
    inside = (numer/denom) * coef
    return inside.cos().pow(2)

# alpha bar in openai paper, which is mean coef squared
def mean_coef_squared(t):
    return cos_term(t)/cos_term(torch.zeros_like(t))

# exp[-int beta] = mean coef squared implies that beta = -(d/dt) log (mean coef squared)
def beta_fn(t):
    numer = t+s
    denom = 1+s
    coef = np.pi / 2
    inside = (numer/denom) * coef
    return inside.tan() * (np.pi / (1+s))

def int_beta_fn(t):
    const = (np.pi / (1+s))
    a = (np.pi / 2) / (1+s)
    b = s
    return const * (-1 / a) * (a*(t+b)).cos().log()

def variance(t):
    return 1 - mean_coef_squared(t)

def std(t):
    return torch.sqrt(variance(t))

# make sure original mean coef and the one derived from beta func equal each other
for tscalar in torch.linspace(0.00001, .9999, 1000):
    t = torch.ones(bsz,) * tscalar
    print(mean_coef_squared(t), (-1.0 * int_beta_fn(t)).exp())

# the math for deriving beta from mean coef squared is below 

'''

alpha bar is mean coef squared. and they set it to alpha bar t = cos_term(t) / cos_term(0). 

# We have that my beta func is, d/dt -log alpha(t) , is 
# d/dt - log cos_term(t)/cos_term(0). 
# = ddt - (log ft - log f0)  
# = ddt + ( log f0 - log ft) 
# =  ddtlog f0 -ddt log ft) 
# = -ddt log ft)

# cos_term(t) = cos( (t+s)/(1+s) * pi/2) ^2 and s=.008

# then we have that beta func is -ddt log cos_term(t) = 

\begin{align}
&-\frac{d}{dt} \log \cos\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)^2\\
&=-\Bigg[\frac{1}{\cos\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)^2} \Bigg]
\Bigg[
\frac{d}{dt} \cos\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)^2
\Bigg]\\
&=-\Bigg[\frac{1}{\cos\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)^2} \Bigg]
\Bigg[
2\cos\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)
\Bigg]
\Bigg[
\frac{d}{dt} \cos\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)
\Bigg]\\
&=-2\Bigg[\frac{\cos\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)}{\cos\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)^2} \Bigg]
\Bigg[
\frac{d}{dt} \cos\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)
\Bigg]\\
&=-2\Bigg[\frac{1}{\cos\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)} \Bigg]
\Bigg[
\frac{d}{dt} \cos\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)
\Bigg]\\
&=-2\Bigg[\frac{1}{\cos\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)} \Bigg]
\Bigg[
-\sin\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)
\Bigg]
\frac{d}{dt} \Bigg(\frac{t+s}{1+s} \frac{\pi}{2}\Bigg)
\\
&=-2\Bigg[\frac{1}{\cos\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)} \Bigg]
\Bigg[
-\sin\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)
\Bigg]
\Bigg(\frac{\pi}{2} \frac{1}{1+s}\frac{d}{dt}(t+s)\Bigg)
\\
&=-2\Bigg[\frac{1}{\cos\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)} \Bigg]
\Bigg[
-\sin\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)
\Bigg]
\Bigg(\frac{\pi}{2} \frac{1}{1+s}\Bigg)
\\
&=2\Bigg[\frac{\sin\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)}{\cos\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)} \Bigg]
\Bigg(\frac{\pi}{2} \frac{1}{1+s}\Bigg)
\\
&=\Bigg[\frac{\sin\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)}{\cos\Big(\frac{t+s}{1+s} * \frac{\pi}{2}\Big)} \Bigg]
\frac{\pi}{1+s}
\\
&=\tan\Bigg(\frac{t+s}{1+s} * \frac{\pi}{2}\Bigg)
\frac{\pi}{1+s}
\\
\end{align}

Then, you do integration by substitution

# our integrand is

def beta_fn(t):
    numer = t+s
    denom = 1+s
    coef = np.pi / 2
    inside = (numer/denom) * coef
    return inside.tan() * (np.pi / (1+s))

# use integration by substitution by noting that beta is a const times tan of a linear function
# use that tan = sin / cos
# use that int of 1/u du is log u 
# c*tan(a(x+b))=c * sin(a(x+b))/cos(a(x+b)). 
# take c out of the integral.
# let u = cos(a(x+b)). then  
# du = - asin(a(x+b)) = du -> -du/a sin(ax)dx
# int tan (a(x+b)) dx 
# = int sin(a(x+b)) / cos (a(x+b)) dx 
# = int sin(a(x+b))  / u    dx
# =int - 1/a   u      du
# = -1/a int  1/u du
# = -1/a log (u)
# = -1/a log(cos((ax+b)))
# now bring c back in.

def int_beta_fn(t):
    const = (np.pi / (1+s))
    a = (np.pi / 2) / (1+s)
    b = s
    return const * (-1 / a) * (a*(t+b)).cos().log()

'''



