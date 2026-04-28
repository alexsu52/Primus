import os
import glob
import subprocess
import re

def main():
    # Find all files starting with 'log_'
    log_files = glob.glob('log_*')
    # Sort files alphabetically
    log_files.sort()

    python_env = "/home/amd/asuslov/env/bin/python"
    parse_script = "/home/amd/asuslov/projects/experiments/mbzuai/results/parse_log.py"

    # Print table header
    header_format = "{:<120} | {:<15} | {:<16} | {:<20}"
    print(header_format.format("Log File Name", "Avg TFLOP/s/GPU", "Avg Tokens/s/GPU", "Avg Memory Usage"))
    print("-" * 180)

    for log_file in log_files:
        if not os.path.isfile(log_file):
            continue

        # Run the parse script
        try:
            result = subprocess.run(
                [python_env, parse_script, log_file],
                capture_output=True,
                text=True,
                check=False
            )
            output = result.stdout + result.stderr
        except Exception as e:
            output = ""

        # Parse Avg TFLOP/s/GPU
        # Example line: Throughput (TFLOP/s/GPU) - Max: 349.5, Min: 178.2, Avg: 333.47
        tflop_match = re.search(r'Throughput \(TFLOP/s/GPU\).*?Avg:\s*([\d\.]+)', output)
        tflop_val = tflop_match.group(1) if tflop_match else "OOM"

        # Parse Avg Tokens/s/GPU
        # Example line: Tokens (tokens/s/GPU) - Max: 1007.0, Min: 513.3, Avg: 960.72
        tokens_match = re.search(r'Tokens \(tokens/s/GPU\).*?Avg:\s*([\d\.]+)', output)
        tokens_val = tokens_match.group(1) if tokens_match else "OOM"

        # Parse Avg Memory Usage
        # Example line: Avg ROCm Memory Usage: 131.68 GB (68.59%)
        mem_match = re.search(r'Avg ROCm Memory Usage:\s*(.*)', output)
        if not mem_match:
            mem_match = re.search(r'Avg HIP Memory Usage:\s*(.*)', output)
        mem_val = mem_match.group(1).strip() if mem_match else "N/A"

        print(header_format.format(log_file, tflop_val, tokens_val, mem_val))

if __name__ == "__main__":
    main()
