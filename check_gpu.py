import torch

# Print version of PyTorch, cuda and cudnn
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'CUDNN version: {torch.backends.cudnn.version()}')

def check_gpu_availability():
    # Check if CUDA (GPU support) is available in PyTorch
    if torch.cuda.is_available():
        # Print the number of GPUs available
        print(f'Number of GPUs available: {torch.cuda.device_count()}')
        # Print the name of each GPU
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        return True
    else:
        print('No GPU available, using CPU instead.')
        return False

# Run the check
check_gpu_availability()
