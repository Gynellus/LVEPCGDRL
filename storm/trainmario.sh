env_name=SuperMarioBrosProgressiveStages-v3
python -u trainmario.py \
    -n "${env_name}-run2-fromScratch" \
    -seed 1 \
    -config_path "config_files/STORM.yaml" \
    -env_name "${env_name}" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" 