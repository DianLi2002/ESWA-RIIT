import argparse
import tester

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Saving and loading parameters
    parser.add_argument('--flag', type=str, default='test', help='train or test')
    parser.add_argument('--save_test_folder', type=str, default='test_results', help='Folder to save test results')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop size')
    # GPU parameters
    parser.add_argument('--multi_gpu', type=bool, default=False, help='Whether to use multiple GPUs')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Enable cudnn benchmark')
    # Testing parameters
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker threads for data loading')

    parser.add_argument('--gpu_id', type=int, default=0, help='IDs of GPU to use')

    parser.add_argument('--dataset_name', type=str, default='BJTU', help='Dataset name: BJTU / BJTU_E / ICVL')
    parser.add_argument('--merged_root', type=str, default='',
                        help='Root of the cached dataset. If empty, the default path hard-coded in tester.py is used')
    parser.add_argument('--in_channels', type=int, default=31, help='Number of input channels for the generator')
    parser.add_argument('--out_channels', type=int, default=31, help='Number of output channels for the generator')

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
   
    parser.add_argument('--method_name', type=str, default='riit', help='Name of the method to use')
    parser.add_argument('--ex_name', type=str, default='_dim64', help='Name of the method to use')
    parser.add_argument('--dim', type=int, default=64, help='dim')
    
    parser.add_argument('--model_path', type=str, default='checkpoints/riit_dim64_bjtu.pth', help='Path to the model to be tested')

    opt = parser.parse_args()
    tester.Tester(opt)