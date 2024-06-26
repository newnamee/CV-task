from simplecv import apex_ddp_train as train
from data import isaid
from module import farseg
import torch

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # 加速
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    # 设置随机种子
    args = train.parser.parse_args()
    # 调用训练函数，传入参数，执行训练
    train.run(local_rank=args.local_rank,
              config_path=args.config_path,
              model_dir=args.model_dir,
              opt_level=args.opt_level,
              cpu_mode=args.cpu,
              after_construct_launcher_callbacks=[],
              opts=args.opts)
