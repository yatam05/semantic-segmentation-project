import torch
import yaml
import cv2
import os
from models.model import initialize_model
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class InferenceModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]

def build_engine(model):
    dummy = torch.randn(input_shape).cuda()
    torch.onnx.export(model, dummy, ONNX_PATH, opset_version=11, input_names=["input"], output_names=["output"])
    
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 24)
    
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(ONNX_PATH, "rb") as f:
        parser.parse(f.read())

    engine = builder.build_engine(network, config)

    with open(ENGINE_PATH, "wb") as f:
        f.write(engine.serialize())
    print(f"Saved engine to ", ENGINE_PATH)
    
def process_frame(frame):
    image = cv2.resize(frame, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - norm_mean) / norm_std
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image, dtype=np.float32)
    image = np.expand_dims(image, 0)
    return image    

def display_results(image, pred, fps, ms):
    image = np.transpose(image, (1,2,0))
    image = image * norm_std + norm_mean
    image = np.clip(image, 0, 1)
    colors = plt.cm.get_cmap('tab20', num_classes)
    colored_mask = colors(pred / num_classes)[..., :3]

    combined = np.hstack((image, colored_mask))
    combined = cv2.resize(combined, (1024, 600))
    cv2.putText(combined, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined, f"Inference Time: {ms:.1f} ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Captured Image | Predicted Mask", combined)
    cv2.waitKey(1)

norm_mean = [0.485,0.456,0.406]
norm_std = [0.229,0.224,0.225]

ONNX_PATH = 'checkpoints/model.onnx'
ENGINE_PATH = 'checkpoints/model.engine'

with open("config.yml") as f:
    config = yaml.safe_load(f)
num_classes = config["dataset"]["num_classes"]
image_size = config["dataset"]["image_size"]

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std)
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = initialize_model("Testing", num_classes, None, None, None, device)
model = InferenceModel(model)
model.aux_classifier = torch.nn.Identity()

input_shape = (1, 3, image_size, image_size)
output_shape = (1, num_classes, image_size, image_size)
h_input = np.zeros(input_shape, dtype=np.float32)
h_output = np.zeros(output_shape, dtype=np.float32)

logger = trt.Logger()

if not os.path.exists(ENGINE_PATH):
    print("Engine not found. Building TensorRT engine...")
    build_engine(model)

with open(ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
input_index = engine.get_binding_index("input")
output_index = engine.get_binding_index("output")
context.set_binding_shape(input_index, input_shape)

d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)

bindings = [0] * engine.num_bindings
bindings[input_index] = int(d_input)
bindings[output_index] = int(d_output)

stream = cuda.Stream()

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error: could not open camera.")
    exit

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = cuda.Event()
    end = cuda.Event()
    start.record(stream)

    h_input = process_frame(frame)
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)

    end.record(stream)
    stream.synchronize()

    ms = start.time_till(end)
    fps = 1000.0 / ms
 
    output = h_output.reshape(output_shape)
    pred_tensor = np.argmax(output, axis=1)
    display_results(h_input[0], pred_tensor[0], fps, ms)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
