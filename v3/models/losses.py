import torch
import torch.nn.functional as F
import numpy as np

def entropy_loss(x):
    """
    Returns the entropy of the distribution parameterized by x
    Since the distributions are assumed multivariate normal, x represents the variances.
    """
    sum_log_x = torch.sum(torch.log(x), dim=-1)
    d = x.size(-1)
    constant_term = d * np.log(2 * np.pi) + 1
    entropy = 0.5 * (sum_log_x + constant_term)
    return entropy

def threshed_jsd_loss(p, q, s, thresh):
    """
    Returns the Jensen-Shannon Divergence loss between the two logits p and q.
    If the similarity s is less than the threshold, use the JSD; otherwise, output 0.
    """
    p_prob = F.softmax(p, dim=-1)
    q_prob = F.softmax(q, dim=-1)
    m = (p_prob + q_prob) / 2 + 1e-8  # Avoid log(0)
    log_m = torch.log(m)
    
    # Compute KL divergences using PyTorch's kl_div
    kl_p = F.kl_div(log_m, p_prob, log_target=False, reduction='none').sum(-1)
    kl_q = F.kl_div(log_m, q_prob, log_target=False, reduction='none').sum(-1)
    jsd = 0.5 * (kl_p + kl_q)
    
    mask = (s < thresh).float()
    loss = (jsd * mask).mean()
    return loss

def mi_loss(p, q):
    """
    Returns the Mutual Information MI(p || q) using Kraskov's second approximation.
    """
    k = 3  # Default neighbor count for Kraskov's method
    device = p.device
    N = p.size(0)
    
    # Combine p and q into joint space
    joint = torch.cat([p, q], dim=1)
    
    # Compute pairwise L-infinity distances in joint space
    dist_joint = torch.cdist(joint, joint, p=float('inf'))
    
    # Find k-th nearest neighbor distance (excluding self)
    knn_dist = torch.topk(dist_joint, k + 1, dim=1, largest=False, sorted=True)[0]
    epsilon = knn_dist[:, k]
    
    # Compute distances in p and q spaces
    dist_p = torch.cdist(p, p, p=float('inf'))
    dist_q = torch.cdist(q, q, p=float('inf'))
    
    # Count neighbors within epsilon for each point
    mask_p = (dist_p <= epsilon.view(-1, 1)).float()
    mask_p -= torch.eye(N, device=device)
    n_p = mask_p.sum(dim=1)
    
    mask_q = (dist_q <= epsilon.view(-1, 1)).float()
    mask_q -= torch.eye(N, device=device)
    n_q = mask_q.sum(dim=1)
    
    # Compute mutual information using digamma function
    psi = torch.digamma
    mi = psi(torch.tensor(k, device=device)) - (psi(n_p + 1).mean() + psi(n_q + 1).mean()) + psi(torch.tensor(N, device=device))
    
    return mi
