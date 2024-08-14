import requests
from PIL import Image
from transformers import AutoImageProcessor
import torch

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
# print(f"Image file format : {image.format}")
# print(f"Image size : {image.size}")
# print(f"Image colormode : {image.mode}")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
image = torch.randn(3, 612, 415)

# 0~1 사이로 정규화
image = (image - image.min()) / (image.max() - image.min())

# 0~255 사이로 스케일링
image = image * 255

image2 = torch.randn(3, 400, 100)

# 0~1 사이로 정규화
image2 = (image2 - image2.min()) / (image2.max() - image2.min())

# 0~255 사이로 스케일링
image2 = image2 * 255

# 정수형으로 변환
image = image.to(torch.uint8)
image2 = image2.to(torch.uint8)
images = [image, image2]
inputs = processor(images=images, return_tensors="pt")

print(inputs)