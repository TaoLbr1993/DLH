import struct
import sys
import math

THRESH = 1e-3  # 允许的偏差，例如 |norm-1| <= 1e-3 视为已归一化

def read_fvecs_meta(path):
    with open(path, "rb") as f:
        dim_bytes = f.read(4)
        if not dim_bytes:
            raise ValueError("文件为空")
        d = struct.unpack("i", dim_bytes)[0]
        f.seek(0, 2)
        fsize = f.tell()
    rec_bytes = 4 + 4 * d
    if fsize % rec_bytes != 0:
        raise ValueError("文件大小与维度不匹配")
    n = fsize // rec_bytes
    return d, n, rec_bytes

def main(path):
    d, n, rec_bytes = read_fvecs_meta(path)
    min_norm = float("inf")
    max_norm = 0.0
    sum_norm = 0.0
    with open(path, "rb") as f:
        for i in range(n):
            buf = f.read(rec_bytes)
            if len(buf) != rec_bytes:
                break
            dim = struct.unpack_from("i", buf, 0)[0]
            assert dim == d
            vec = struct.unpack_from(f"{d}f", buf, 4)
            norm = math.sqrt(sum(v * v for v in vec))
            min_norm = min(min_norm, norm)
            max_norm = max(max_norm, norm)
            sum_norm += norm
    avg_norm = sum_norm / n
    print(f"path={path}")
    print(f"n={n}, d={d}")
    print(f"norm: min={min_norm:.6f}, max={max_norm:.6f}, avg={avg_norm:.6f}")
    ok = abs(avg_norm - 1) <= THRESH and abs(max_norm - 1) <= THRESH and abs(min_norm - 1) <= THRESH
    print("已归一化" if ok else "未归一化或偏差较大")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"用法: python {sys.argv[0]} /path/to/file.fvecs")
        sys.exit(1)
    main(sys.argv[1])