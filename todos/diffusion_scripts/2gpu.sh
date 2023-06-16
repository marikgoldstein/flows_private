python -m torch.distributed.run --master_port=29501 --nnodes=1 --nproc_per_node=2 train.py --index 0 --do_resume 0 --debug 1 --dataset mnist                      

