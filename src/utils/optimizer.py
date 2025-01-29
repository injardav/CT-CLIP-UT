from torch.optim import AdamW, Adam


def separate_params_by_weight_decay(params):
    """
    Separate parameters into two groups: 
    - `wd_params` for those eligible for weight decay.
    - `no_wd_params` for those that are not (e.g., biases, batch norm).
    """
    wd_params = [p for p in params if p.ndim >= 2]
    no_wd_params = [p for p in params if p.ndim < 2]
    return wd_params, no_wd_params

def get_optimizer(
    params,
    lr=1e-4,
    wd=1e-4,
    betas=(0.9, 0.99),
    eps=1e-8,
    filter_requires_grad=False,
    group_wd_params=True,
    **kwargs
):
    """
    Create an optimizer for training, supporting grouping by weight decay.

    Args:
        params (iterable): Model parameters.
        lr (float): Learning rate.
        wd (float): Weight decay (L2 regularization). Default: 1e-4.
        betas (tuple): Coefficients for Adam's beta parameters. Default: (0.9, 0.99).
        eps (float): Epsilon value for numerical stability. Default: 1e-8.
        filter_requires_grad (bool): Whether to filter out parameters without gradients.
        group_wd_params (bool): Whether to separate parameters into weight decayable and non-decayable groups.

    Returns:
        torch.optim.Optimizer: Configured optimizer (Adam or AdamW).
    """
    if filter_requires_grad:
        params = [p for p in params if p.requires_grad]

    if wd == 0:
        # Use Adam if no weight decay is applied
        return Adam(params, lr=lr, betas=betas, eps=eps)

    if group_wd_params:
        # Separate weight decayable and non-decayable parameters
        wd_params, no_wd_params = separate_params_by_weight_decay(params)
        params = [
            {'params': wd_params}, 
            {'params': no_wd_params, 'weight_decay': 0}
        ]

    return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
