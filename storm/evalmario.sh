env_name=SuperMarioBrosProgressiveStages-v3
python -u evalmario.py \
    -env_name "${env_name}" \
    -run_name "${env_name}-run1-fromScratch"\
    -config_path "config_files/STORM.yaml" 
