import re
import sys
from collections import defaultdict

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <log_file>")
        sys.exit(1)

    log_file = sys.argv[1]
    rank_to_layers = defaultdict(list)

    # Pattern to match rank and layer_idx ignoring ANSI escape codes
    pattern = re.compile(r'rank-(\d+).*?layer_idx:\s*(\d+)\s*is in recompute_layer_ids')

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    rank = int(match.group(1))
                    layer_idx = int(match.group(2))
                    rank_to_layers[rank].append(layer_idx)
    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found.")
        sys.exit(1)

    if not rank_to_layers:
        print("No matching log entries found.")
        return

    for rank in sorted(rank_to_layers.keys()):
        layers = rank_to_layers[rank]
        # remove duplicates while preserving order
        unique_layers = list(dict.fromkeys(layers))
        print(f"Rank {rank}: {unique_layers}")

if __name__ == "__main__":
    main()
