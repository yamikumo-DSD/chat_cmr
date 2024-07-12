def suggest_device() -> str:
    """
    Returns:
        str: "cuda", "mps", or "cpu"
    """
    import torch
    if torch.cuda.is_available():  
        return "cuda"
    else: 
        if (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            return "mps"
        else:
            return "cpu"