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
    
    padding =  (0, 0, 0, 0, 1, 0)
    
    match mode:
        case 'cumsum':            
            dA_cumsum = F.pad(dA[:, 1:], padding).cumsum(1).exp()
            x = dB_u / (dA_cumsum + 1e-12)
            x = x.cumsum(1) * dA_cumsum
            y = torch.einsum('bldn,bln->bld', x, C)
        
        case 'logcumsumexp':  # more numerically stable (Heisen sequence)
            dB_u_log = complex_log(dB_u)
            dA_star = F.pad(dA[:, 1:].cumsum(1), padding)
            x_log = torch.logcumsumexp(dB_u_log - dA_star, 1) + dA_star
            y = torch.einsum('bldn,bln->bld', x_log.real.exp() * torch.cos(x_log.imag), C)
            
    return y + u * D


# the mismatch between the cumsum and logcumsumexp modes will grow quickly as sequence length scales up
if __name__ == "__main__":
    for length in [4, 8, 16, 32, 64, 128, 256]:
        u = -1 + 2 * torch.rand(2, length, 32)
        dt = torch.ones(2, length, 32)
        A =  -torch.rand(32, 16)
        B = torch.rand(2, length, 16)
        C = torch.rand(2, length, 16)
        D = torch.rand(32)
        
        output_cumsum = selective_scan(u, dt, A, B, C, D, mode='cumsum')
        output_logcumsumexp = selective_scan(u, dt, A, B, C, D, mode='logcumsumexp')
    
        print(f"mismatch at length {length} is {(output_cumsum - output_logcumsumexp).abs().max()}")
    