import os
from loss import char_sam_loss

def create_generator(opt):
    # Use method_name (default 'sit') to select the generator method
    method = opt.method_name.lower()
    if opt.flag == 'train':
        if method == 'riit':
            from model.RIIT import RIIT
            generator = RIIT(inp_channels=opt.in_channels, out_channels=opt.out_channels, dim=opt.dim, num_blocks=[2, 2, 2])
            loss_function = char_sam_loss.PhysicsInformedLoss(lambda_sam=opt.lambda_sam)
        else:
            raise NotImplementedError(f"Generator method '{opt.method_name}' is not implemented.")      

        return generator, loss_function
    
    elif opt.flag == 'test':
        if method == 'riit':
            from model.RIIT import RIIT
            generator = RIIT(inp_channels=opt.in_channels, out_channels=opt.out_channels, dim=opt.dim, num_blocks=[2, 2, 2])
        else:
            raise NotImplementedError(f"Generator method '{opt.method_name}' is not implemented.")      

        return generator

    else:
        raise ValueError(f"Invalid flag '{opt.flag}'. Expected 'train' or 'test'.")

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)