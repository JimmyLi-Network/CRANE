import ctypes
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from crane.bridge import (
    ANEBridgeError,
    ANEBridgeLibrary,
    DEFAULT_ANE_BRIDGE_PATH,
    _build_dummy_weight_blob,
    _free_bridge_blob,
    build_baked_linear_mil,
    _unpack_dyn_matmul_output,
    build_dyn_matmul_mil,
    pack_baked_linear_input,
    pack_dyn_matmul_input,
)

_DYN_MATMUL_KERNEL_CACHE: dict[tuple[str, int, int, int], "ANEKernel"] = {}
_STATIC_LINEAR_KERNEL_CACHE: dict[tuple[str, str], "ANEKernel"] = {}


def _safe_weight_filename(logical_kernel_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in logical_kernel_name)


@dataclass(frozen=True)
class ANECompiledKernelSpec:
    mil_text: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[np.dtype, ...]
    output_dtypes: tuple[np.dtype, ...]


class ANEKernel:
    def __init__(
        self,
        *,
        bridge: ANEBridgeLibrary,
        kernel_handle,
        input_shapes: Sequence[Sequence[int]],
        output_shapes: Sequence[Sequence[int]],
        input_dtypes: Sequence[np.dtype],
        output_dtypes: Sequence[np.dtype],
    ) -> None:
        self.bridge = bridge
        self.kernel_handle = kernel_handle
        self.input_shapes = tuple(tuple(int(v) for v in shape) for shape in input_shapes)
        self.output_shapes = tuple(tuple(int(v) for v in shape) for shape in output_shapes)
        self.input_dtypes = tuple(np.dtype(dtype) for dtype in input_dtypes)
        self.output_dtypes = tuple(np.dtype(dtype) for dtype in output_dtypes)

    @classmethod
    def compile(
        cls,
        *,
        mil_text: str,
        input_shapes: Sequence[Sequence[int]],
        output_shapes: Sequence[Sequence[int]],
        input_dtypes: Sequence[np.dtype],
        output_dtypes: Sequence[np.dtype],
        bridge_path: str = DEFAULT_ANE_BRIDGE_PATH,
    ) -> "ANEKernel":
        bridge = ANEBridgeLibrary(bridge_path)
        input_sizes = (ctypes.c_size_t * len(input_shapes))(
            *[int(np.prod(shape)) * np.dtype(dtype).itemsize for shape, dtype in zip(input_shapes, input_dtypes)]
        )
        output_sizes = (ctypes.c_size_t * len(output_shapes))(
            *[int(np.prod(shape)) * np.dtype(dtype).itemsize for shape, dtype in zip(output_shapes, output_dtypes)]
        )
        mil_bytes = mil_text.encode("utf-8")
        dummy_blob_ptr, dummy_blob_len = _build_dummy_weight_blob(bridge)
        kernel_handle = bridge.lib.ane_bridge_compile(
            mil_bytes,
            len(mil_bytes),
            dummy_blob_ptr,
            dummy_blob_len,
            len(input_shapes),
            input_sizes,
            len(output_shapes),
            output_sizes,
        )
        _free_bridge_blob(dummy_blob_ptr)
        if not kernel_handle:
            raise ANEBridgeError("ane_bridge_compile failed")
        return cls(
            bridge=bridge,
            kernel_handle=kernel_handle,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
        )

    @classmethod
    def compile_multi_weights(
        cls,
        *,
        mil_text: str,
        weight_blobs: dict[str, bytes],
        input_shapes: Sequence[Sequence[int]],
        output_shapes: Sequence[Sequence[int]],
        input_dtypes: Sequence[np.dtype],
        output_dtypes: Sequence[np.dtype],
        bridge_path: str = DEFAULT_ANE_BRIDGE_PATH,
    ) -> "ANEKernel":
        bridge = ANEBridgeLibrary(bridge_path)
        input_sizes = (ctypes.c_size_t * len(input_shapes))(
            *[int(np.prod(shape)) * np.dtype(dtype).itemsize for shape, dtype in zip(input_shapes, input_dtypes)]
        )
        output_sizes = (ctypes.c_size_t * len(output_shapes))(
            *[int(np.prod(shape)) * np.dtype(dtype).itemsize for shape, dtype in zip(output_shapes, output_dtypes)]
        )
        mil_bytes = mil_text.encode("utf-8")
        n_weights = len(weight_blobs)
        weight_names = (ctypes.c_char_p * n_weights)(*[
            name.encode("utf-8") for name in weight_blobs
        ])
        weight_buffers = [ctypes.create_string_buffer(blob) for blob in weight_blobs.values()]
        weight_datas = (ctypes.c_void_p * n_weights)(*[
            ctypes.cast(buf, ctypes.c_void_p) for buf in weight_buffers
        ])
        weight_lens = (ctypes.c_size_t * n_weights)(*[
            len(blob) for blob in weight_blobs.values()
        ])
        kernel_handle = bridge.lib.ane_bridge_compile_multi_weights(
            mil_bytes,
            len(mil_bytes),
            weight_names,
            weight_datas,
            weight_lens,
            n_weights,
            len(input_shapes),
            input_sizes,
            len(output_shapes),
            output_sizes,
        )
        if not kernel_handle:
            raise ANEBridgeError("ane_bridge_compile_multi_weights failed")
        return cls(
            bridge=bridge,
            kernel_handle=kernel_handle,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
        )

    def run(
        self,
        inputs: Sequence[np.ndarray],
        *,
        output_buffers: Sequence[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        if len(inputs) != len(self.input_shapes):
            raise ValueError("number of input tensors does not match compiled kernel")

        prepared_inputs: list[np.ndarray] = []
        for array, shape, dtype in zip(inputs, self.input_shapes, self.input_dtypes):
            prepared = np.ascontiguousarray(np.asarray(array, dtype=dtype))
            if prepared.shape != shape:
                raise ValueError(f"expected input shape {shape}, got {prepared.shape}")
            prepared_inputs.append(prepared)

        if output_buffers is None:
            prepared_outputs = [
                np.empty(shape, dtype=dtype) for shape, dtype in zip(self.output_shapes, self.output_dtypes)
            ]
        else:
            if len(output_buffers) != len(self.output_shapes):
                raise ValueError("number of output buffers does not match compiled kernel")
            prepared_outputs = []
            for array, shape, dtype in zip(output_buffers, self.output_shapes, self.output_dtypes):
                prepared = np.ascontiguousarray(np.asarray(array, dtype=dtype))
                if prepared.shape != shape:
                    raise ValueError(f"expected output shape {shape}, got {prepared.shape}")
                prepared_outputs.append(prepared)

        for idx, array in enumerate(prepared_inputs):
            self.bridge.lib.ane_bridge_write_input(
                self.kernel_handle,
                idx,
                array.ctypes.data_as(ctypes.c_void_p),
                array.nbytes,
            )

        if not self.bridge.lib.ane_bridge_eval(self.kernel_handle):
            raise ANEBridgeError("ane_bridge_eval failed")

        for idx, array in enumerate(prepared_outputs):
            self.bridge.lib.ane_bridge_read_output(
                self.kernel_handle,
                idx,
                array.ctypes.data_as(ctypes.c_void_p),
                array.nbytes,
            )
        return prepared_outputs

    def close(self) -> None:
        if self.kernel_handle is not None:
            self.bridge.lib.ane_bridge_free(self.kernel_handle)
            self.kernel_handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def compile_dyn_matmul_kernel(
    *,
    ic: int,
    oc: int,
    seq: int,
    bridge_path: str = DEFAULT_ANE_BRIDGE_PATH,
) -> ANEKernel:
    key = (bridge_path, int(seq), int(ic), int(oc))
    kernel = _DYN_MATMUL_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = ANEKernel.compile(
            mil_text=build_dyn_matmul_mil(ic=ic, oc=oc, seq=seq),
            input_shapes=((1, ic, 1, seq + oc),),
            output_shapes=((1, oc, 1, seq),),
            input_dtypes=(np.float32,),
            output_dtypes=(np.float32,),
            bridge_path=bridge_path,
        )
        _DYN_MATMUL_KERNEL_CACHE[key] = kernel
    return kernel


def compile_baked_linear_kernel(
    *,
    ic: int,
    oc: int,
    seq: int,
    logical_kernel_name: str,
    weights: np.ndarray,
    bridge_path: str = DEFAULT_ANE_BRIDGE_PATH,
) -> ANEKernel:
    key = (bridge_path, logical_kernel_name)
    kernel = _STATIC_LINEAR_KERNEL_CACHE.get(key)
    if kernel is None:
        weight_array = np.asarray(weights, dtype=np.float32)
        if weight_array.shape != (ic, oc):
            raise ValueError(f"expected weights shape {(ic, oc)}, got {weight_array.shape}")
        bridge = ANEBridgeLibrary(bridge_path)
        out_len = ctypes.c_size_t()
        transposed = np.ascontiguousarray(weight_array.T)
        blob_ptr = bridge.lib.ane_bridge_build_weight_blob(
            transposed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(oc),
            int(ic),
            ctypes.byref(out_len),
        )
        if not blob_ptr:
            raise ANEBridgeError("ane_bridge_build_weight_blob failed")
        try:
            weight_blob = ctypes.string_at(blob_ptr, out_len.value)
        finally:
            _free_bridge_blob(blob_ptr)
        weight_path = f"@model_path/weights/{_safe_weight_filename(logical_kernel_name)}.bin"
        kernel = ANEKernel.compile_multi_weights(
            mil_text=build_baked_linear_mil(ic=ic, oc=oc, seq=seq, weight_path=weight_path),
            weight_blobs={weight_path: weight_blob},
            input_shapes=((1, ic, 1, seq),),
            output_shapes=((1, oc, 1, seq),),
            input_dtypes=(np.float32,),
            output_dtypes=(np.float32,),
            bridge_path=bridge_path,
        )
        _STATIC_LINEAR_KERNEL_CACHE[key] = kernel
    return kernel


def run_dyn_matmul_kernel(
    kernel: ANEKernel,
    activations: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    packed_input = np.ascontiguousarray(
        pack_dyn_matmul_input(
            np.asarray(activations, dtype=np.float32),
            np.asarray(weights, dtype=np.float32),
        )
    )
    output = kernel.run([packed_input])[0]
    return _unpack_dyn_matmul_output(output)


def run_baked_linear_kernel(
    kernel: ANEKernel,
    activations: np.ndarray,
) -> np.ndarray:
    packed_input = np.ascontiguousarray(pack_baked_linear_input(np.asarray(activations, dtype=np.float32)))
    output = kernel.run([packed_input])[0]
    return _unpack_dyn_matmul_output(output)
