import sys
from pathlib import Path

def main(path: str, want: int):
    n = sum(1 for ln in Path(path).read_text().splitlines() if ln.strip())
    print(f"{path}: rows={n} want={want}")
    raise SystemExit(0 if n == want else 1)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
