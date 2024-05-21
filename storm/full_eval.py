import subprocess

# List of levels
levels = [
    "1-1", "1-2", "1-3", "2-1", "3-1", "3-3", "4-1", "4-2",
    "5-1", "5-3", "6-1", "6-2", "6-3", "7-1", "8-1"
]

# Base script template with shebang
script_template = """#!/bin/bash
env_name={env_name}
python -u evalmario.py \\
    -env_name "${{env_name}}" \\
    -run_name "SuperMarioBrosRandomStages-v3-life_done-wm_2L512D8H-100k-seed1" \\
    -config_path "config_files/STORM.yaml"
"""

# Iterate over each level and run the script
for level in levels:
    
    env_name = f"SuperMarioBros-{level}-v3"
    script_content = script_template.format(env_name=env_name)

    # Write the script content to a temporary file
    with open('temp_script.sh', 'w') as file:
        file.write(script_content)

    # Make the script executable
    subprocess.run(['chmod', '+x', 'temp_script.sh'])

    # Run the script
    subprocess.run(['./temp_script.sh'])

    # Clean up the temporary script file
    subprocess.run(['rm', 'temp_script.sh'])
