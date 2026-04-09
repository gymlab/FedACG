import math
import hashlib
import torch
import numpy as np


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 1000003


class LayerWiseQJL:
    def __init__(
        self,
        device="cuda",
        qjl_ratio=0.5,
        use_orthogonal=True,
        seed=0,
        skip_small_tensors=True,
        small_tensor_threshold=256,
        block_size=2048,
        min_m=32,
        max_m=256,
    ):
        self.device = torch.device(device)
        self.qjl_ratio = qjl_ratio
        self.use_orthogonal = use_orthogonal
        self.seed = seed
        self.S_cache = {}
        self.sqrt_pi_2 = math.sqrt(math.pi / 2.0)
        self.skip_small_tensors = skip_small_tensors
        self.small_tensor_threshold = small_tensor_threshold
        self.block_size = block_size
        self.min_m = min_m
        self.max_m = max_m

    def _resolve_m(self, d_block: int) -> int:
        m = max(self.min_m, int(d_block * self.qjl_ratio))
        m = min(m, self.max_m, d_block)
        return max(1, m)

    def _build_projection(self, d: int, m: int, key: str, dtype=torch.float32):
        local_seed = self.seed + stable_hash(f"{key}_{d}_{m}")
        g = torch.Generator(device=self.device)
        g.manual_seed(local_seed)

        if self.use_orthogonal:
            A = torch.randn(m, d, generator=g, device=self.device, dtype=dtype)
            Q, _ = torch.linalg.qr(A.t())
            S = Q.t() * math.sqrt(d)
        else:
            S = torch.randn(m, d, generator=g, device=self.device, dtype=dtype)

        return S

    def get_projection(self, cache_key: str, d: int, dtype=torch.float32):
        m = self._resolve_m(d)
        key = (cache_key, d, m, str(dtype), str(self.device))
        if key not in self.S_cache:
            self.S_cache[key] = self._build_projection(d, m, cache_key, dtype=dtype)
        return self.S_cache[key], m

    def _split_blocks(self, x: torch.Tensor):
        n = x.numel()
        for start in range(0, n, self.block_size):
            end = min(start + self.block_size, n)
            yield start, end, x[start:end]

    def compress_for_similarity(self, param_key: str, delta: torch.Tensor):
        """
        유사도 추정용: sign bits + norm만 반환
        실제 delta는 별도로 raw 전송
        """
        d = delta.numel()

        if self.skip_small_tensors and d <= self.small_tensor_threshold:
            return None  # 작은 레이어는 유사도 추정 스킵

        delta_flat = delta.reshape(-1).to(self.device, dtype=torch.float32)
        blocks = []

        for block_idx, (start, end, block) in enumerate(self._split_blocks(delta_flat)):
            d_block = block.numel()
            block_key = f"{param_key}_block{block_idx}"
            S, m = self.get_projection(block_key, d_block)

            projected = torch.matmul(S, block)
            sign_bits = (projected >= 0).cpu().numpy().astype(np.bool_)
            packed_uint8 = np.packbits(sign_bits, bitorder="little")
            norm_val = torch.norm(block, p=2).item()

            blocks.append({
                "start": start,
                "end": end,
                "d_block": d_block,
                "m_block": m,
                "packed_sign": torch.from_numpy(packed_uint8).to(torch.uint8),
                "norm": norm_val,
            })

        return blocks  # None이 아니면 유사도 추정 가능

    def estimate_inner_product(self, param_key: str, query: torch.Tensor, compressed_blocks):
        """
        ProdQJL(query, delta) 추정
        query: 서버의 g_global (raw)
        compressed_blocks: 클라이언트가 보낸 sign bits + norm
        """
        if compressed_blocks is None:
            return None

        query_flat = query.reshape(-1).to(self.device, dtype=torch.float32)
        total_ip = 0.0
        total_m = 0

        for block_idx, block_info in enumerate(compressed_blocks):
            start = block_info["start"]
            end = block_info["end"]
            d_block = block_info["d_block"]
            m_block = block_info["m_block"]
            norm_val = block_info["norm"]

            block_key = f"{param_key}_block{block_idx}"
            S, _ = self.get_projection(block_key, d_block)

            # query block에 S 적용
            query_block = query_flat[start:end]
            Sq = torch.matmul(S, query_block)  # [m]

            # sign bits 복원
            packed_np = block_info["packed_sign"].cpu().numpy()
            unpacked = np.unpackbits(packed_np, bitorder="little")[:m_block]
            sign_vec = torch.from_numpy(unpacked).to(self.device, dtype=torch.float32)
            sign_vec = sign_vec.mul_(2.0).sub_(1.0)

            # ProdQJL = (√π/2 / m) * norm * <Sq, sign(Sk)>
            ip_block = self.sqrt_pi_2 / m_block * norm_val * torch.dot(Sq, sign_vec).item()
            total_ip += ip_block
            # total_m += 1

        # return total_ip / total_m  # 블록별 평균
        return total_ip
    
def build_qjl_helper(args, device):
    return LayerWiseQJL(
        device=device,
        qjl_ratio=getattr(args.server, "qjl_ratio", 1.0),
        use_orthogonal=getattr(args.server, "use_orthogonal", True),
        seed=getattr(args, "seed", 0),
        skip_small_tensors=getattr(args.server, "skip_small_tensors", True),
        small_tensor_threshold=getattr(args.server, "small_tensor_threshold", 256),
        block_size=getattr(args.server, "block_size", 2048),
        min_m=getattr(args.server, "min_m", 32),
        max_m=getattr(args.server, "max_m", 256),
    )