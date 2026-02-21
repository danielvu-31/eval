from huggingface_hub import snapshot_download

# Download only the benchmark folder
snapshot_download(
    repo_id="MizzenAI/HPDv3",
    repo_type="dataset",
    allow_patterns="benchmark/*",
    local_dir="/lustre/scratch/client/movian/research/users/kiennt104/CrossDistill/HPDv3"
)