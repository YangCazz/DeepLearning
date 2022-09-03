import os

import torch
import torch.distributed as dist

# 初始化各进程的基础参数
def init_distributed_mode(args):
    # os.environ["xx"]根据一个字符串映射到系统环境的一个对象
    #
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])                 # 进程号
        args.world_size = int(os.environ['WORLD_SIZE'])     # 分配大小
        args.gpu = int(os.environ['LOCAL_RANK'])            # 分配GPU号
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])         #
        args.gpu = args.rank % torch.cuda.device_count()    # 多重任务分配
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True                                 # 多GPU训练模式打开
    torch.cuda.set_device(args.gpu)                         # 将模型和数据加载到对应GPU上
    args.dist_backend = 'nccl'                              # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    # 定义多进程的基本参数
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

# 清空线程
def cleanup():
    dist.destroy_process_group()

# 检查
def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

# 获得分配内存大小
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

# 获取GPU号
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

# 判别是否是主进程
def is_main_process():
    return get_rank() == 0

#
def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value