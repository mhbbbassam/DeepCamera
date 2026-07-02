"""
Model loader wrapper — tries to load optimized model according to env_config.HardwareEnv.
Returns an object with a standardized __call__(image_or_path, conf=...) -> list of detections
Each detection is dict: {"x1":..., "y1":..., "x2":..., "y2":..., "conf":..., "cls": int}
"""

import os
from pathlib import Path
import numpy as np

# Reuse the project's env_config if available
try:
    from lib.env_config import HardwareEnv
except Exception:
    # fallback local import
    from skills.lib.env_config import HardwareEnv

def _to_standard_boxes_from_ultralytics_result(ultra_result):
    """Convert ultralytics result object to list of dicts."""
    boxes = []
    try:
        data = ultra_result.boxes.data.cpu().numpy()
        for row in data:
            x1, y1, x2, y2, conf, cls = row.tolist()
            boxes.append({"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "conf": float(conf), "cls": int(cls)})
    except Exception:
        pass
    return boxes

class ModelWrapper:
    def __init__(self, loader_type, impl):
        self.loader_type = loader_type
        self.impl = impl

    def __call__(self, image_or_path, conf=0.5, classes=None):
        """
        Must return list of dicts: {"x1","y1","x2","y2","conf","cls"}
        """
        # ultralytics YOLO
        if self.loader_type == "ultralytics":
            results = self.impl(image_or_path, conf=conf, classes=classes, verbose=False)
            if not results:
                return []
            return _to_standard_boxes_from_ultralytics_result(results[0])
        # onnx/coreml wrappers should provide same interface
        if self.loader_type == "onnx_coreml":
            return self.impl(image_or_path, conf=conf)
        if self.loader_type == "tensorrt":
            return self.impl(image_or_path, conf=conf)
        # pytorch fallback (ultralytics handles .pt too)
        return []

def load_model(cfg: dict):
    """
    Heuristic loader:
      - If .trt/.engine exists, try to load via TensorRT Python API or wrapper
      - Else if onnx exists, try onnxruntime (possibly with TRT EP)
      - Else fallback to ultralytics YOLO(.pt/.onnx supported)
    """
    model_path = cfg.get("model_path")
    model_format = cfg.get("model_format", "auto")
    env = HardwareEnv.detect()

    p = Path(model_path)
    # Prefer TensorRT engine if explicitly requested or file endswith .trt/.engine
    try:
        if model_format in ("tensorrt", "auto") and p.suffix.lower() in (".trt", ".engine"):
            # Attempt to load via TensorRT (user must have tensorrt python API)
            try:
                import tensorrt as trt  # noqa
                # Implement engine load wrapper here or call a helper (left as TODO)
                def trt_infer(img_path, conf=0.5):
                    # TODO: implement actual inference using TensorRT engine
                    return []
                return ModelWrapper("tensorrt", trt_infer)
            except ImportError:
                pass

        # ONNX path
        if model_format in ("onnx", "auto") and p.suffix.lower() == ".onnx":
            try:
                import onnxruntime as ort
                # Prefer TRT EP if available in providers (deployment dependent)
                providers = None
                try:
                    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                    sess = ort.InferenceSession(str(p), providers=providers)
                    # implement wrapper similar to env_config._OnnxCoreMLModel
                    def onnx_wrapper(img_path, conf=0.5):
                        # TODO: implement onnx preprocessing + session.run + decode
                        return []
                    return ModelWrapper("onnx_coreml", onnx_wrapper)
                except Exception:
                    # fallback to CPU EP
                    sess = ort.InferenceSession(str(p), providers=['CPUExecutionProvider'])
                    def onnx_wrapper_cpu(img_path, conf=0.5):
                        return []
                    return ModelWrapper("onnx_coreml", onnx_wrapper_cpu)
            except ImportError:
                pass

        # Fallback: let ultralytics handle .pt/.onnx and many cases
        try:
            from ultralytics import YOLO
            model = YOLO(str(p))
            return ModelWrapper("ultralytics", model)
        except Exception:
            raise

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    raise RuntimeError("No suitable model loader found")
