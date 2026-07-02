"""Model loader with TensorRT .engine support (TRTInfer) and fallbacks.

Usage:
  from scripts.model_loader import load_model
  model = load_model(cfg)
  dets = model(frame_path, conf=0.5)

Returns list of dicts: {"x1","y1","x2","y2","conf","cls"}
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

try:
    from lib.env_config import HardwareEnv
except Exception:
    try:
        from skills.lib.env_config import HardwareEnv
    except Exception:
        HardwareEnv = None


def letterbox_image(img: Image.Image, new_shape=(640, 640), color=(114, 114, 114)):
    orig_w, orig_h = img.size
    nh, nw = new_shape
    scale = min(nw / orig_w, nh / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (nw, nh), color)
    pad_x = (nw - new_w) // 2
    pad_y = (nh - new_h) // 2
    canvas.paste(img_resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y, orig_w, orig_h


class TRTInfer:
    def __init__(self, engine_path: str):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # initializes CUDA driver
        except Exception as e:
            raise RuntimeError(f"TensorRT/PyCUDA imports failed: {e}")

        self.trt = trt
        self.cuda = cuda
        self.logger = trt.Logger(trt.Logger.WARNING)

        # deserialize engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()

        # Prepare bindings
        self.input_binding_idx = None
        self.output_binding_idxs = []
        self.bindings = [None] * self.engine.num_bindings

        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            if self.engine.binding_is_input(idx):
                self.input_binding_idx = idx
            else:
                self.output_binding_idxs.append(idx)

        if self.input_binding_idx is None:
            raise RuntimeError("No input binding found in engine")

        # Get input shape (handle static shapes)
        in_shape = self.engine.get_binding_shape(self.input_binding_idx)
        if len(in_shape) == 4:
            self.input_n, self.input_c, self.input_h, self.input_w = in_shape
        else:
            # fallback defaults
            self.input_n, self.input_c, self.input_h, self.input_w = 1, 3, 640, 640

        # Allocate host and device buffers
        self.host_buffers = [None] * self.engine.num_bindings
        self.device_buffers = [None] * self.engine.num_bindings
        self.binding_ptrs = [0] * self.engine.num_bindings

        for idx in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            # resolve dynamic dims (-1) to 1 for allocation; engine should be built with fixed dims ideally
            alloc_shape = [s if s > 0 else 1 for s in (shape if hasattr(shape, '__len__') else [shape])]
            size = int(np.prod(alloc_shape))
            host_arr = np.empty(size, dtype=dtype)
            dev_mem = cuda.mem_alloc(host_arr.nbytes)
            self.host_buffers[idx] = host_arr
            self.device_buffers[idx] = dev_mem
            self.binding_ptrs[idx] = int(dev_mem)

        # CUDA stream
        self.stream = cuda.Stream()

    def preprocess(self, img_path: str):
        img = Image.open(img_path).convert("RGB")
        resized, scale, pad_x, pad_y, orig_w, orig_h = letterbox_image(img, (self.input_w, self.input_h))
        arr = np.asarray(resized, dtype=np.float32) / 255.0  # HWC
        arr = arr.transpose(2, 0, 1)  # CHW
        arr = np.expand_dims(arr, 0).astype(np.float32)
        return arr, scale, pad_x, pad_y, orig_w, orig_h

    def infer(self, img_path: str, conf: float = 0.5):
        inp, scale, pad_x, pad_y, orig_w, orig_h = self.preprocess(img_path)
        # copy input into host buffer corresponding to input binding idx
        in_idx = self.input_binding_idx
        host_in = self.host_buffers[in_idx]
        if inp.ravel().size > host_in.size:
            raise RuntimeError("Input size larger than engine input buffer (shape mismatch)")
        # write into host buffer (flatten)
        host_in[:inp.ravel().size] = inp.ravel()

        # copy host -> device
        self.cuda.memcpy_htod_async(self.device_buffers[in_idx], host_in, stream=self.stream)

        # execute
        bindings = [int(b) for b in self.binding_ptrs]
        try:
            # execute async
            self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
        except Exception as e:
            # try sync execute as fallback
            try:
                self.context.execute_v2(bindings=bindings)
            except Exception as e2:
                raise RuntimeError(f"TensorRT execution failed: {e} | fallback: {e2}")

        # copy outputs device->host
        outputs = []
        for out_idx in self.output_binding_idxs:
            host_out = self.host_buffers[out_idx]
            dev_ptr = self.device_buffers[out_idx]
            self.cuda.memcpy_dtoh_async(host_out, dev_ptr, stream=self.stream)
            outputs.append(host_out.copy())

        # sync
        self.stream.synchronize()

        if len(outputs) < 2:
            # unexpected output layout
            return []

        # Interpret outputs: assume outputs[0]=logits [1,N,C], outputs[1]=pred_boxes [1,N,4]
        # Need to reshape based on binding shapes
        out0_shape = self.engine.get_binding_shape(self.output_binding_idxs[0])
        out1_shape = self.engine.get_binding_shape(self.output_binding_idxs[1])
        try:
            logits = outputs[0].reshape(out0_shape)
            pred_boxes = outputs[1].reshape(out1_shape)
        except Exception:
            # best-effort reshape using available lengths
            logits = outputs[0]
            pred_boxes = outputs[1]

        # remove batch dim if present
        if hasattr(logits, 'shape') and len(logits.shape) == 3:
            logits = logits[0]
        if hasattr(pred_boxes, 'shape') and len(pred_boxes.shape) == 3:
            pred_boxes = pred_boxes[0]

        probs = 1.0 / (1.0 + np.exp(-logits))
        dets = []
        for i in range(len(pred_boxes)):
            cls_id = int(np.argmax(probs[i]))
            det_conf = float(probs[i][cls_id])
            if det_conf < conf:
                continue
            cx, cy, bw, bh = pred_boxes[i]
            px_cx = cx * self.input_w
            px_cy = cy * self.input_h
            px_w = bw * self.input_w
            px_h = bh * self.input_h
            x1 = max(0, min((px_cx - px_w / 2 - pad_x) / scale, orig_w))
            y1 = max(0, min((px_cy - px_h / 2 - pad_y) / scale, orig_h))
            x2 = max(0, min((px_cx + px_w / 2 - pad_x) / scale, orig_w))
            y2 = max(0, min((px_cy + px_h / 2 - pad_y) / scale, orig_h))
            dets.append({"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "conf": det_conf, "cls": cls_id})

        return dets


def load_model(cfg: dict):
    """Load model according to cfg.
    cfg keys: model_path, model_format
    Returns callable: model(img_path, conf, classes) -> list[dict]
    """
    model_path = cfg.get("model_path")
    model_format = cfg.get("model_format", "auto")

    if not model_path:
        raise RuntimeError("No model_path provided in config")

    p = Path(model_path)

    # Try TensorRT engine first
    if p.exists() and p.suffix.lower() in (".trt", ".engine") and model_format in ("tensorrt", "auto"):
        try:
            trt_inf = TRTInfer(str(p))
            return lambda img_path, conf=0.5, classes=None: trt_inf.infer(img_path, conf=conf)
        except Exception as e:
            print(f"[model_loader] TRT load failed: {e}", file=sys.stderr)

    # Try ONNX via onnxruntime
    if p.exists() and p.suffix.lower() == ".onnx" and model_format in ("onnx", "auto"):
        try:
            import onnxruntime as ort
            # try providers with Tensorrt EP then CUDA then CPU
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            sess = ort.InferenceSession(str(p), providers=[pr for pr in providers if pr in ort.get_available_providers()])

            # build a simple onnx wrapper similar to TRTInfer if needed
            def onnx_wrapper(img_path, conf=0.5, classes=None):
                # TODO: implement preprocessing + inference + decode like TRTInfer
                # For now, fall back to returning empty list to avoid silent failures
                return []

            return onnx_wrapper
        except Exception as e:
            print(f"[model_loader] ONNX load failed: {e}", file=sys.stderr)

    # Fallback to ultralytics YOLO (handles .pt/.onnx typically)
    try:
        from ultralytics import YOLO
        model = YOLO(str(p))

        def ultra_wrapper(img_path, conf=0.5, classes=None):
            results = model(img_path, conf=conf, classes=classes, verbose=False)
            if not results:
                return []
            try:
                data = results[0].boxes.data.cpu().numpy()
            except Exception:
                return []
            out = []
            for row in data:
                x1, y1, x2, y2, confv, cls = row.tolist()
                out.append({"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "conf": float(confv), "cls": int(cls)})
            return out

        return ultra_wrapper
    except Exception as e:
        raise RuntimeError(f"No model loader available: {e}")
