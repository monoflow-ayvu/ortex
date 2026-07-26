import importlib.util
from pathlib import Path

import torch
from torchvision.models import ResNet50_Weights, resnet50

if importlib.util.find_spec("onnx") is None:
    raise SystemExit("torch.onnx.export needs onnx: python3 -m pip install onnx")

# The dynamo exporter needs onnxscript; without it fall back to the legacy tracer.
dynamo = importlib.util.find_spec("onnxscript") is not None
if not dynamo:
    print("onnxscript not available, exporting with the legacy ONNX exporter")

models_dir = Path(__file__).resolve().parents[1] / "models"
models_dir.mkdir(exist_ok=True)
output_path = models_dir / "resnet50.onnx"

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()
onnx_input = torch.randn(1, 3, 224, 224)

with torch.inference_mode():
    torch.onnx.export(
        model,
        (onnx_input,),
        output_path,
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        export_params=True,
        opset_version=19,
        dynamo=dynamo,
    )

print(f"Wrote {output_path}")
