env_name=SuperMarioBrosRandomStages-v3
python -u evalmario.py \
    -env_name "${env_name}" \
    -run_name "${env_name}-life_done-wm_2L512D8H-100k-seed1"\
    -config_path "config_files/STORM.yaml" 
