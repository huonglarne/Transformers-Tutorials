from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch

from PIL import Image
import requests

feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
image

pixel_values = feature_extractor(image, return_tensors="pt").pixel_values

with torch.no_grad():
  outputs = model(pixel_values)
  predicted_depth = outputs.predicted_depth