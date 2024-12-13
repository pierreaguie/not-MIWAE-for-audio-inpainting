import torch


def soft_clipping(x : torch.Tensor, W : float, b : float) -> torch.Tensor:
    """ 
    Returns the missing mask for the input tensor using a soft clipping model:
        P(s = 1 | x) = sigmoid(-W * (|x| - b))

    Inputs:
    ----------------
    - x (torch.Tensor): input tensor
    - W (float): multiplicative parameter of the soft clipping model
    - b (float): threshold

    Outputs:
    ----------------
    - s (torch.Tensor): missing mask
    """
    
    logits = -W * (torch.abs(x) - b)
    
    p = torch.nn.functional.sigmoid(logits)
    
    s = torch.bernoulli(p)
    return s



def oneway_soft_clipping(x : torch.Tensor, W : float, b : float) -> torch.Tensor:
    """ 
    Returns the missing mask for the input tensor using a soft clipping model:
        P(s = 1 | x) = sigmoid(-W * (x - b))

    Inputs:
    ----------------
    - x (torch.Tensor): input tensor
    - W (float): multiplicative parameter of the soft clipping model
    - b (float): threshold

    Outputs:
    ----------------
    - s (torch.Tensor): missing mask
    """
    
    logits = -W * (x - b)
    p = torch.nn.functional.sigmoid(logits)
    s = torch.bernoulli(p)
    return s


def hard_clipping(x : torch.Tensor, threshold : float) -> torch.Tensor:
    """ 
    Returns the missing mask for the input tensor using a hard clipping model:
        s = 1 if |x| > threshold else 0
    
    Inputs:
    ----------------
    - x (torch.Tensor): input tensor
    - threshold (float): threshold

    Outputs:
    ----------------
    - s (torch.Tensor): missing mask
    """
    s = (torch.sign(torch.abs(x) - threshold) + 1) / 2
    return s


def normalize(x : torch.Tensor) -> torch.Tensor:
    """ 
    Normalizes an input waveform x to have a maximal absolute value of 1.
    """
    return x / torch.max(torch.abs(x))
