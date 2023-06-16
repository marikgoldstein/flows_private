from models_openai import get_openai_mnist_unet
from models_bigunet import get_big_unet
from models_dhariwal import get_dhariwal_unet
from models_tiny import get_tiny_net
from models_lucid import get_lucid_unet
from models_lucid2 import get_lucid2_unet
from models_scoreunet import get_scoreunet
from models_dit import get_dit

def get_unet_fn(arch):

    if arch == 'tiny':
        return get_tiny_net
    elif arch == 'scoreunet':
        return get_scoreunet
    elif arch == 'lucid':
        return get_lucid_unet
    elif arch == 'lucid2':
        return get_lucid2_unet
    elif arch == 'openai':
        return get_openai_mnist_unet
    elif arch == 'bigunet':
        return get_big_unet
    elif arch == 'dhariwal':
        return get_dhariwal_unet
    elif arch == 'dit':
        return get_dit
    else:
        assert False

