import argparse
import trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Saving and loading parameters
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--flag', type=str, default='train', help='train or test')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Directory to save the model checkpoints')
    parser.add_argument('--save_val_folder', type=str, default='val_results', help='Folder to save validation results')
    parser.add_argument('--save_by_epoch', type=int, default=1, help='Save model every specified number of epochs')
    parser.add_argument('--save_test_folder', type=str, default='test_results', help='Folder to save test results')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop size')
    # GPU parameters
    parser.add_argument('--multi_gpu', type=bool, default=False, help='Whether to use multiple GPUs')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Enable cudnn benchmark')
    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--b1', type=float, default=0.9, help='Adam beta1 parameter')
    parser.add_argument('--b2', type=float, default=0.999, help='Adam beta2 parameter')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for the optimizer')
    parser.add_argument('--lr_decrease_epoch', type=int, default=200, help='Decrease learning rate every specified number of epochs')
    parser.add_argument('--lr_decrease_factor', type=float, default=0.5, help='Learning rate decrease factor')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker threads for data loading')
    # parameters
    parser.add_argument('--gpu_ids', type=str, default='0', help='IDs of GPUs to use')

    parser.add_argument('--dataset_name', type=str, default='BJTU', help='Dataset name: BJTU / BJTU_E / ICVL')
    parser.add_argument('--merged_root', type=str, default='',
                        help='Root of the cached dataset. If empty, the default path hard-coded in trainer.py is used')
    parser.add_argument('--in_channels', type=int, default=31, help='Number of input channels for the generator')
    parser.add_argument('--out_channels', type=int, default=31, help='Number of output channels for the generator')

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--loss_function', type=str, default='char_sam_loss', help='Name of the loss function to use')
    parser.add_argument('--lambda_sam', type=float, default=1.0, help='lambda for the physics-informed loss')
    
    parser.add_argument('--method_name', type=str, default='riit', help='Name of the method to use')
    parser.add_argument('--ex_name', type=str, default='_dim64', help='Name of the ex_name')
    parser.add_argument('--dim', type=int, default='64', help='dim')

    opt = parser.parse_args()
    
    trainer.Trainer(opt)