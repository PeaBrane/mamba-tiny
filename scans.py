import torch
from torch.nn import functional as F


def complex_log(input, eps=1e-12):
    eps = input.new_tensor(eps)
    real = input.abs().maximum(eps).log()
    imag = (input < 0).to(input.dtype) * torch.pi
    return torch.complex(real, imag)


def selective_scan(u, dt, A, B, C, D, mode='logcumsumexp'):
    dA = torch.einsum('bld,dn->bldn', dt, A)
    dB_u = torch.einsum('bld,bld,bln->bldn', dt, u, B)
    dA = dA.clamp(min=-20)
    
    match mode:
        case 'cumsum':
            dA_cumsum = F.pad(dA[:, 1:], (0, 0, 0, 0, 0, 1)).flip(1).cumsum(1).exp().flip(1)
            x = dB_u * dA_cumsum
            x = x.cumsum(1) / (dA_cumsum + 1e-12)
            y = torch.einsum('bldn,bln->bld', x, C)
        
            return y + u * D
        
        case 'logcumsumexp':
            dB_u_log = complex_log(dB_u)
            
            dA_star = F.pad(dA[:, 1:].cumsum(1), (0, 0, 0, 0, 1, 0))
            x_log = torch.logcumsumexp(dB_u_log - dA_star, 1) + dA_star
            
            y = torch.einsum('bldn,bln->bld', x_log.real.exp() * torch.cos(x_log.imag), C)
            return y + u * D
