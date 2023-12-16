
import netron
import torch

from torchvision.models import resnet50


def netron_model(model, input_shape, onnx_path=None, port=1234):
    dummy_input = torch.randn(*input_shape)
    # 类名.随机名字.onnx
    onnx_path = f"{model.__class__.__name__}.{torch.rand(1).item():.6f}.pt" if onnx_path is None else onnx_path
    script_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(script_model, onnx_path)
    netron.start(onnx_path, address=("localhost", port))
    print(f"Model visualized at http://localhost:{port}")

if __name__ == "__main__":
    model = resnet50()
    netron_model(model, (1, 3, 224, 224))