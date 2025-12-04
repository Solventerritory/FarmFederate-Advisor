# hardware/backend_integration/upload_test_image.py
import requests
url = "http://localhost:8000/predict"
files = {'image': open('some_image.jpg','rb')}
data = {'text':'(no text)','sensors':'SENSORS: soil_moisture=20%, temp=30'}
r = requests.post(url, files=files, data=data)
print(r.status_code, r.text)
