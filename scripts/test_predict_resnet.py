import io
from PIL import Image
import requests

img=Image.new('RGB',(224,224),(128,128,128))
buf=io.BytesIO()
img.save(buf,'JPEG')
buf.seek(0)
files={'image':('test.jpg',buf,'image/jpeg')}
resp=requests.post('http://127.0.0.1:5000/predict',files=files,data={'model':'resnet50'})
print('Status',resp.status_code)
print('Body',resp.text)
