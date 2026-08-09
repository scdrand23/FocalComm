import os
import torch
import torch.distributed as dist


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


# def init_distributed_mode(args):
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ['WORLD_SIZE'])
#         args.gpu = int(os.environ['LOCAL_RANK'])
#     elif 'SLURM_PROCID' in os.environ:
#         args.rank = int(os.environ['SLURM_PROCID'])
#         args.gpu = args.rank % torch.cuda.device_count()
#     else:
#         print('Not using distributed mode')
#         args.distributed = False
#         return

#     args.distributed = True

#     torch.cuda.set_device(args.gpu)
#     args.dist_backend = 'nccl'
#     print('| distributed init (rank {}): {}'.format(
#         args.rank, args.dist_url), flush=True)
#     torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
#                                          world_size=args.world_size, rank=args.rank)
#     torch.distributed.barrier()
#     setup_for_distributed(args.rank == 0)

def init_distributed_mode(args):
    """
    Initialize distributed training mode with better SLURM support
    """
    # Check if CUDA is available first
    if not torch.cuda.is_available():
        print('CUDA not available. Running on CPU.')
        args.distributed = False
        args.gpu = None
        return
    
    # Handle torchrun/torch.distributed.launch environment variables
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        
    # Handle SLURM environment variables
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ.get('SLURM_NPROCS', '1'))
        
        # Better GPU assignment for SLURM
        if 'SLURM_LOCALID' in os.environ:
            # Use SLURM_LOCALID for GPU assignment
            args.gpu = int(os.environ['SLURM_LOCALID'])
        else:
            # Fallback to modulo assignment
            num_gpus = torch.cuda.device_count()
            args.gpu = args.rank % num_gpus
            
        # Print debug info
        print(f'SLURM mode: rank={args.rank}, world_size={args.world_size}, gpu={args.gpu}')
        print(f'Available GPUs: {torch.cuda.device_count()}')
        
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.gpu = 0 if torch.cuda.is_available() else None
        return

    # Validate GPU assignment
    if args.gpu >= torch.cuda.device_count():
        print(f'Warning: GPU {args.gpu} not available. Using GPU 0.')
        args.gpu = 0
    
    args.distributed = True
    
    # Set CUDA device with error handling
    try:
        torch.cuda.set_device(args.gpu)
        print(f'Successfully set CUDA device to GPU {args.gpu}')
    except RuntimeError as e:
        print(f'Error setting CUDA device: {e}')
        print('Attempting to initialize without explicit device setting...')
        # Don't set device explicitly, let PyTorch handle it
        pass
    
    # Set up distributed backend
    args.dist_backend = 'nccl'
    
    # Handle dist_url
    if not hasattr(args, 'dist_url') or args.dist_url == 'env://':
        # For SLURM, construct the URL from MASTER_ADDR and MASTER_PORT
        if 'MASTER_ADDR' in os.environ and 'MASTER_PORT' in os.environ:
            args.dist_url = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        else:
            # Default to env://
            args.dist_url = 'env://'
    
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    
    try:
        torch.distributed.init_process_group(
            backend=args.dist_backend, 
            init_method=args.dist_url,
            world_size=args.world_size, 
            rank=args.rank
        )
        torch.distributed.barrier()
        setup_for_distributed(args.rank == 0)
        print(f'Successfully initialized distributed training on rank {args.rank}')
    except Exception as e:
        print(f'Error initializing distributed training: {e}')
        args.distributed = False
        raise

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

