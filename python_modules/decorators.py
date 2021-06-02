

def loss_to_value(func):
    def wrapper(*args, **kwargs):
        loss = func(*args, **kwargs)
        return loss.item()
    return wrapper
