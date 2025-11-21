import os
import numpy as np
from django.core.files.storage import default_storage
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from django.views.decorators.csrf import csrf_exempt

# Load Model
model = load_model(r'C:\Users\jhaaa\Downloads\Heart_Data\heart2.h5')

# Prediction Function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    result = "Normal" if prediction < 0.5 else "Diseased"

    return result, img

# View Function
@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        img_file = request.FILES['image']
        img_path = default_storage.save('tmp/' + img_file.name, img_file)

        result, img = predict_image(img_path)

        # Convert image for display
        buffer = BytesIO()
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        encoded_img = base64.b64encode(buffer.read()).decode('utf-8')

        return render(request, 'result.html', {"result": result, "image_data": encoded_img})

    return render(request, 'upload.html')
