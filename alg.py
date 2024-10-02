import torch
import time
import warnings
def eoft_WO(batch_size,M0, eps, r, c , break_err, num_iters, a_samples, b_samples, d0):
    K = torch.exp((-M0)/(eps)).to(torch.float64).to('cuda')
    n_samples = (K.shape[1])
    r_c = torch.zeros(batch_size, n_samples)
    r = torch.tensor(r)
    c = torch.tensor(c)
    q = torch.ones(batch_size, n_samples)                    ##initailize the q vector
    v = (1 / a_samples ) * torch.ones(batch_size, n_samples) ##initailize the v vector   
    u = (1 / a_samples ) * torch.ones(batch_size, n_samples) ##initailize the u vector
    d = torch.ones(batch_size, n_samples) * d0

    q[:, :a_samples] = r
    r_c[:, :a_samples] = r
    r_c[:, a_samples + b_samples:] = c
    r_c[:, a_samples + b_samples:] = -r_c[:, a_samples + b_samples:]
    r_c = r_c.to(torch.float64).to('cuda')
    v = v.to(torch.float64).to('cuda')
    u = u.to(torch.float64).to('cuda')
    q = q.to(torch.float64).to('cuda') 
    d = d.to(torch.float64).to('cuda') ## the virtual self flow
    
    diag_indices = torch.arange(n_samples, device=K.device)
    batch_indices = torch.arange(batch_size, device=K.device)[:, None]
    
    t1 =time.time()
    pow_rc = (torch.pow(r_c,2)/4.0).to('cuda')
    pre_q = ((r_c / 2.0) - d ).to('cuda')
    
    for i in range(num_iters):
        down_u = ((K * v.unsqueeze(1)).sum(2))
        u = (q + d )/down_u
        down_v = ((torch.transpose(K,2,1) * u.unsqueeze(1)).sum(2))
        v = (q - r_c + d )/down_v
        h = d / (u * v)
        ##the updata of K
        K[batch_indices, diag_indices, diag_indices] = h
        inner_sum = ((K * v.unsqueeze(1)).sum(2)) * ((torch.transpose(K, 2, 1) * u.unsqueeze(1)).sum(2))
        q = pre_q + torch.sqrt(inner_sum + pow_rc)
        
        if i % 100 == 0:    ##print the error every 100 iterations
            P = ((u.unsqueeze(2)) * K) * (v.unsqueeze(1))
            err = torch.mean(torch.abs((P.sum(2) - P.sum(1)) - r_c)) 
            print(f'it is {i} iterations and loss is {err}')

        if err <  break_err:
            break

        if (torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)) or torch.any(torch.isnan(q)) or torch.any(torch.isinf(q))  ):
            warnings.warn('--------------------------------------Warning: numerical errors at iteration %d' % i)
            break    
    
    P = ((u.unsqueeze(2)) * K) * (v.unsqueeze(1))
    t2 = time.time()
    time_eoft = t2 -t1
    
    P_constraint = P.sum(2) - P.sum(1)
    P_constraint = torch.mean(torch.abs(P_constraint - r_c),dim=(1))
    print(f'iterations end with {i} ,maximum iterations is  {num_iters} 轮 精度为{err} and all time is {t2 - t1}')

    return P , time_eoft


def eoft_EN(batch_size, M0, eps, r, c , break_err, num_iters, a_samples, b_samples, d0, Edge_C = None, Node_C=None):
    K = torch.exp((-M0)/(eps)).to(torch.float64).to('cuda')
    n_samples = K.shape[1]
    q = torch.zeros(batch_size,n_samples)
    v = (1 / a_samples) * torch.ones(batch_size,n_samples)      
    u = (1 / b_samples) * torch.ones(batch_size,n_samples)     
    d = torch.ones(batch_size,n_samples) * d0

    r_c = torch.zeros(batch_size, n_samples)
    r = torch.tensor(r)
    c = torch.tensor(c)
    q[:, :a_samples] = r
    r_c[:, :a_samples] = r
    r_c[:, a_samples + b_samples:] = c
    r_c[:, a_samples + b_samples:] = -r_c[:, a_samples + b_samples:]

    v = v.to(torch.float64).to('cuda')
    u = u.to(torch.float64).to('cuda')
    q = q.to(torch.float64).to('cuda')
    d = d.to(torch.float64).to('cuda')
    r_c = r_c.to(torch.float64).to('cuda')
    t1 =time.time()
    pow_rc = (torch.pow(r_c,2)/4.0).to('cuda')
    pre_q = ((r_c / 2.0) - d ).to('cuda')
    if Node_C != None:
        array_C =  torch.ones_like(q) * Node_C 
    Edge_Matrix = Edge_C.to('cuda')

    for i in range(num_iters):
        down_u = ((K * v.unsqueeze(1)).sum(2))
        u = (q + d )/down_u
        down_v = ((torch.transpose(K,2,1) * u.unsqueeze(1)).sum(2))
        v = (q - r_c + d )/down_v
        
        ##tthe update of K(for warm up, we only update K after 20 iterations)
        if Edge_C != None and i > 20:
            u_recip = (1 / u).unsqueeze(1) 
            v_recip = (1 / v).unsqueeze(2) 
            log_input = u_recip * Edge_Matrix * v_recip             
            K = torch.minimum(K, log_input)
            
        inner_sum = ((K * v.unsqueeze(1)).sum(2)) * ((torch.transpose(K, 2, 1) * u.unsqueeze(1)).sum(2))
        q = pre_q + torch.sqrt(inner_sum + pow_rc)

        if Node_C != None and i > 20 :
            q = torch.minimum(q, array_C)

        if i % 100 == 0 :
            P = ((u.unsqueeze(2)) * K) * (v.unsqueeze(1))
            err = torch.mean(torch.abs((P.sum(2) - P.sum(1)) - r_c)) 
            print(f'it is {i} iterations and loss is {err}')

        if err <  break_err:
            break

        if (torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)) or torch.any(torch.isnan(q)) or torch.any(torch.isinf(q))  ):
            warnings.warn('--------------------------------------Warning: numerical errors at iteration %d' % i)
            failed = True
            break

    P = ((u.unsqueeze(2)) * K) * (v.unsqueeze(1))
    t2 = time.time()
    # print(f'P 的运算时间为{t9-t8}')
    P_constraint = P.sum(2) - P.sum(1)
    P_constraint = torch.mean(torch.abs(P_constraint - r_c),dim=(1))
    time_eoft = t2 -t1
    print(f'iterations end with {i} ,maximum iterations is  {num_iters} 轮 精度为{err} and all time is {t2 - t1}')

    return P , time_eoft
