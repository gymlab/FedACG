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
        qjl_ratio=1.0,
        use_orthogonal=False,
        seed=0,
        skip_small_tensors=True,
        small_tensor_threshold=256,
        block_size=2048,              # 핵심 추가
        min_m=32,                     # 너무 작은 m 방지
        max_m=256,                    # m 상한
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
            # block-wise에서만 제한적으로 사용 권장
            A = torch.randn(m, d, generator=g, device=self.device, dtype=dtype)
            Q, _ = torch.linalg.qr(A.t())   # [d, m]
            S = Q.t()                       # [m, d]
        else:
            S = torch.randn(m, d, generator=g, device=self.device, dtype=dtype)

        return S

    def get_projection(self, cache_key_prefix: str, d: int, dtype=torch.float32):
        m = self._resolve_m(d)
        cache_key = (cache_key_prefix, d, m, str(dtype), str(self.device))

        if cache_key not in self.S_cache:
            self.S_cache[cache_key] = self._build_projection(d, m, cache_key_prefix, dtype=dtype)

        return self.S_cache[cache_key], m

    def _split_blocks(self, x: torch.Tensor):
        blocks = []
        n = x.numel()
        for start in range(0, n, self.block_size):
            end = min(start + self.block_size, n)
            blocks.append((start, end, x[start:end]))
        return blocks

    def compress(self, param_key: str, delta: torch.Tensor):
        d = delta.numel()

        if self.skip_small_tensors and d <= self.small_tensor_threshold:
            return ("raw", delta.detach().clone(), delta.shape)

        delta_flat = delta.reshape(-1).to(self.device, dtype=torch.float32)
        blocks = self._split_blocks(delta_flat)

        packed_blocks = []
        for block_idx, (start, end, block) in enumerate(blocks):
            d_block = block.numel()
            block_key = f"{param_key}_block{block_idx}"

            S, m = self.get_projection(block_key, d_block, dtype=torch.float32)

            projected = torch.matmul(S, block)

            sign_bits_np = (projected >= 0).detach().cpu().numpy().astype(np.bool_)
            packed_uint8_np = np.packbits(sign_bits_np, bitorder="little")

            norm_val = torch.norm(block, p=2).detach().cpu()

            packed_blocks.append({
                "start": start,
                "end": end,
                "d_block": d_block,
                "m_block": m,
                "packed_sign": torch.from_numpy(packed_uint8_np).to(torch.uint8),
                "norm_val": norm_val,
            })

        return ("qjl_block", packed_blocks, delta.shape, d)

    def decompress(self, param_key: str, packed):
        mode = packed[0]

        if mode == "raw":
            _, tensor, orig_shape = packed
            return tensor.reshape(orig_shape)

        elif mode == "qjl_block":
            _, packed_blocks, orig_shape, d_total = packed

            reconstructed_flat = torch.zeros(
                d_total, device=self.device, dtype=torch.float32
            )

            for block_idx, block_info in enumerate(packed_blocks):
                start = block_info["start"]
                end = block_info["end"]
                d_block = block_info["d_block"]
                m_block = block_info["m_block"]
                packed_tensor = block_info["packed_sign"]

                packed_np = packed_tensor.detach().cpu().numpy()
                unpacked_bits = np.unpackbits(packed_np, bitorder="little")[:m_block]

                sign_vec_np = (unpacked_bits.astype(np.float32) * 2.0) - 1.0
                sign_vec = torch.from_numpy(sign_vec_np).to(self.device, dtype=torch.float32)

                norm_val = block_info["norm_val"].to(self.device, dtype=torch.float32)

                block_key = f"{param_key}_block{block_idx}"
                S, _ = self.get_projection(block_key, d_block, dtype=torch.float32)

                coeff = self.sqrt_pi_2 / m_block
                block_rec = coeff * norm_val * torch.matmul(S.t(), sign_vec)
                reconstructed_flat[start:end] = block_rec

            return reconstructed_flat.reshape(orig_shape)