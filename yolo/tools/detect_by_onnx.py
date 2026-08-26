import onnxruntime
import sys
import cv2
import numpy as np
model = sys.argv[1]
from utils.general import non_max_suppression



providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = onnxruntime.InferenceSession(model, providers=providers)
output_names = [x.name for x in session.get_outputs()]
meta = session.get_modelmeta().custom_metadata_map  # metadata
if 'stride' in meta:
    stride, names = int(meta['stride']), eval(meta['names'])

im = cv2.imread(sys.argv[2])
im = im.transpose((2,0,1)).astype(np.float32)
im /= 255.0  # 0 - 255 to 0.0 - 1.0
if len(im.shape) == 3:
    im = im[np.newaxis, :]
input_name = session.get_inputs()[0].name
print(output_names)
y = session.run(output_names, {input_name: im})[0]
pred = non_max_suppression(y)
print(pred)