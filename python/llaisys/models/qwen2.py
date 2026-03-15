from typing import Sequence
from pathlib import Path
import json
import re
import random

import numpy as np
import safetensors

from ctypes import c_int, c_int64, c_size_t, c_void_p, POINTER, byref

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys.models import LlaisysQwen2Meta, LlaisysQwen2Weights
from ..tensor import Tensor


def _numpy_dtype(dtype: DataType):
    if dtype == DataType.F32:
        return np.float32
    if dtype == DataType.F16:
        return np.float16
    if dtype == DataType.BF16:
        return np.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _to_bf16_buffer(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32, order="C")
    bits = arr.view(np.uint32)
    rounding = ((bits >> 16) & 1) + 0x7FFF
    bf = ((bits + rounding) >> 16).astype(np.uint16)
    return bf


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        dtype_str = str(config.get("torch_dtype", "bfloat16")).lower()
        if "bfloat16" in dtype_str:
            dtype = DataType.BF16
        elif "float16" in dtype_str:
            dtype = DataType.F16
        else:
            dtype = DataType.F32

        nlayer = int(config["num_hidden_layers"])
        hs = int(config["hidden_size"])
        nh = int(config["num_attention_heads"])
        nkvh = int(config.get("num_key_value_heads", nh))
        dh = int(config.get("head_dim", hs // nh))
        di = int(config["intermediate_size"])
        maxseq = int(config.get("max_position_embeddings", 2048))
        voc = int(config["vocab_size"])
        epsilon = float(config.get("rms_norm_eps", 1e-5))
        theta = float(config.get("rope_theta", 10000.0))
        eos = config.get("eos_token_id", config.get("eos_token_ids", None))
        if isinstance(eos, list):
            end_token = int(eos[0])
        else:
            end_token = int(eos) if eos is not None else -1

        meta = LlaisysQwen2Meta()
        meta.dtype = dtype
        meta.nlayer = nlayer
        meta.hs = hs
        meta.nh = nh
        meta.nkvh = nkvh
        meta.dh = dh
        meta.di = di
        meta.maxseq = maxseq
        meta.voc = voc
        meta.epsilon = epsilon
        meta.theta = theta
        meta.use_qk_norm = 0
        meta.end_token = end_token

        layer_re = re.compile(r"model\.layers\.(\d+)\.(.+)")
        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            if any(
                name.endswith("self_attn.q_norm.weight") or name.endswith("self_attn.k_norm.weight")
                for name in data_.keys()
            ):
                meta.use_qk_norm = 1
                break

        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta), c_int(device), device_ids, c_int(1)
        )
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents
        self._meta = meta
        self._end_token = end_token

        self._in_embed = Tensor(tensor=self._weights.in_embed, owning=False)
        self._out_embed = Tensor(tensor=self._weights.out_embed, owning=False)
        self._out_norm_w = Tensor(tensor=self._weights.out_norm_w, owning=False)

        self._attn_norm_w = [Tensor(tensor=self._weights.attn_norm_w[i], owning=False) for i in range(nlayer)]
        self._attn_q_w = [Tensor(tensor=self._weights.attn_q_w[i], owning=False) for i in range(nlayer)]
        self._attn_q_b = [Tensor(tensor=self._weights.attn_q_b[i], owning=False) for i in range(nlayer)]
        self._attn_k_w = [Tensor(tensor=self._weights.attn_k_w[i], owning=False) for i in range(nlayer)]
        self._attn_k_b = [Tensor(tensor=self._weights.attn_k_b[i], owning=False) for i in range(nlayer)]
        self._attn_v_w = [Tensor(tensor=self._weights.attn_v_w[i], owning=False) for i in range(nlayer)]
        self._attn_v_b = [Tensor(tensor=self._weights.attn_v_b[i], owning=False) for i in range(nlayer)]
        self._attn_q_norm_w = [
            Tensor(tensor=self._weights.attn_q_norm_w[i], owning=False) if self._weights.attn_q_norm_w[i] else None
            for i in range(nlayer)
        ]
        self._attn_k_norm_w = [
            Tensor(tensor=self._weights.attn_k_norm_w[i], owning=False) if self._weights.attn_k_norm_w[i] else None
            for i in range(nlayer)
        ]
        self._attn_o_w = [Tensor(tensor=self._weights.attn_o_w[i], owning=False) for i in range(nlayer)]
        self._mlp_norm_w = [Tensor(tensor=self._weights.mlp_norm_w[i], owning=False) for i in range(nlayer)]
        self._mlp_gate_w = [Tensor(tensor=self._weights.mlp_gate_w[i], owning=False) for i in range(nlayer)]
        self._mlp_up_w = [Tensor(tensor=self._weights.mlp_up_w[i], owning=False) for i in range(nlayer)]
        self._mlp_down_w = [Tensor(tensor=self._weights.mlp_down_w[i], owning=False) for i in range(nlayer)]

        weight_map = {
            "model.embed_tokens.weight": self._in_embed,
            "lm_head.weight": self._out_embed,
            "model.lm_head.weight": self._out_embed,
            "model.norm.weight": self._out_norm_w,
        }

        np_dtype = _numpy_dtype(dtype)
        embed_cache = None
        out_embed_loaded = False

        use_torch = dtype == DataType.BF16
        for file in sorted(model_path.glob("*.safetensors")):
            if use_torch:
                data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            else:
                data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
            for name_ in data_.keys():
                arr = data_.get_tensor(name_)
                target = None

                if name_ in weight_map:
                    target = weight_map[name_]
                else:
                    match = layer_re.match(name_)
                    if match:
                        idx = int(match.group(1))
                        suffix = match.group(2)
                        if suffix == "input_layernorm.weight":
                            target = self._attn_norm_w[idx]
                        elif suffix == "self_attn.q_proj.weight":
                            target = self._attn_q_w[idx]
                        elif suffix == "self_attn.q_proj.bias":
                            target = self._attn_q_b[idx]
                        elif suffix == "self_attn.k_proj.weight":
                            target = self._attn_k_w[idx]
                        elif suffix == "self_attn.k_proj.bias":
                            target = self._attn_k_b[idx]
                        elif suffix == "self_attn.v_proj.weight":
                            target = self._attn_v_w[idx]
                        elif suffix == "self_attn.v_proj.bias":
                            target = self._attn_v_b[idx]
                        elif suffix == "self_attn.q_norm.weight":
                            target = self._attn_q_norm_w[idx]
                        elif suffix == "self_attn.k_norm.weight":
                            target = self._attn_k_norm_w[idx]
                        elif suffix == "self_attn.o_proj.weight":
                            target = self._attn_o_w[idx]
                        elif suffix == "post_attention_layernorm.weight":
                            target = self._mlp_norm_w[idx]
                        elif suffix == "mlp.gate_proj.weight":
                            target = self._mlp_gate_w[idx]
                        elif suffix == "mlp.up_proj.weight":
                            target = self._mlp_up_w[idx]
                        elif suffix == "mlp.down_proj.weight":
                            target = self._mlp_down_w[idx]

                if target is None:
                    continue

                if use_torch:
                    if dtype == DataType.BF16:
                        arr_np = arr.float().cpu().numpy()
                        buf = _to_bf16_buffer(arr_np)
                        buf = np.ascontiguousarray(buf)
                        target.load(buf.ctypes.data_as(c_void_p))
                        prepared = buf
                    else:
                        arr_np = arr.cpu().numpy()
                        arr_np = np.asarray(arr_np, dtype=np_dtype)
                        arr_np = np.ascontiguousarray(arr_np)
                        target.load(arr_np.ctypes.data_as(c_void_p))
                        prepared = arr_np
                else:
                    arr = np.asarray(arr, dtype=np_dtype)
                    arr = np.ascontiguousarray(arr)
                    target.load(arr.ctypes.data_as(c_void_p))
                    prepared = arr

                if name_ == "model.embed_tokens.weight":
                    embed_cache = prepared
                if name_ in ("lm_head.weight", "model.lm_head.weight"):
                    out_embed_loaded = True

        if not out_embed_loaded and embed_cache is not None:
            self._out_embed.load(embed_cache.ctypes.data_as(c_void_p))

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        seed: int | None = None,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128
        tokens = list(int(t) for t in inputs)
        for next_id in self.generate_stream(
            tokens,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
        ):
            tokens.append(next_id)
        return tokens

    def generate_stream(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        seed: int | None = None,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128
        tokens = list(int(t) for t in inputs)
        rng = random.Random(seed)
        do_sample = temperature > 0.0 and (top_k != 1 or top_p < 1.0)
        for _ in range(max_new_tokens):
            arr = (c_int64 * len(tokens))(*tokens)
            if do_sample:
                next_id = int(
                    LIB_LLAISYS.llaisysQwen2ModelInferSample(
                        self._model,
                        arr,
                        c_size_t(len(tokens)),
                        temperature,
                        int(top_k),
                        top_p,
                        rng.getrandbits(64),
                    )
                )
            else:
                next_id = int(
                    LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, arr, c_size_t(len(tokens)))
                )
            tokens.append(next_id)
            yield next_id
            if self._end_token >= 0 and next_id == self._end_token:
                break
