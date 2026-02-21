
module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate /lustre/scratch/client/movian/research/users/kiennt104/.conda/envs/evaluation+cross


path_ckpt="/lustre/scratch/client/movian/research/users/anhnd72/viet/pretrained_models/model_fid8.1"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 eval_2.py --path_ckpt $path_ckpt