from flask import Flask, request, jsonify
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO
import torch

app = Flask(__name__)

@app.route('/test', methods=['POST'])
def upload_image():

    '''

    Request Example:

        payload = {
            "image": base64_image,
            "metadata": {
                "description": "example image"
            }
        }

        response = requests.post(endpoint_url, json=payload)

    '''

    data = request.json
   
    base64_image = data.get('image')
    image_data = base64.b64decode(base64_image)
   
    image = Image.open(BytesIO(image_data))
   
    opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
   
    glass_model = YOLO('./models/glass.pt')
    drink_model = YOLO('./models/drink.pt')

    glass_results = glass_model.predict(opencvImage, save=False, conf=0.5)
    drink_results = drink_model.predict(opencvImage, save=False, conf=0.75)

    glass_flag = True
    drink_flag = True

    for glass_result, drink_result in zip(glass_results, drink_results):
        try:
            glass_masks = glass_result.masks.data
        except AttributeError:
            glass_flag = False
        else:
            glass_masks = torch.any(glass_masks, dim=0).int() * 255
            glass_masks = glass_masks.cpu().numpy()

        try:
            drink_masks = drink_result.masks.data
        except AttributeError:
            drink_flag = False
        else:
            drink_masks = torch.any(drink_masks, dim=0).int() * 255
            drink_masks = drink_masks.cpu().numpy()



    if glass_flag:
                
        if drink_flag:
            
            glass_masks[drink_masks != 0] = 100

        glass_masks = Image.fromarray(np.uint8(glass_masks)).convert('RGB')
        glass_masks = glass_masks.resize((opencvImage.shape[1], opencvImage.shape[0]))
        glass_masks = np.array(glass_masks)

        for glass_result in glass_results:

            annotator = Annotator(opencvImage)
            
            for box in glass_result.boxes.xyxy.cpu().tolist():
                
                glass = glass_masks[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

                if drink_flag:
                    percentage = f'{len(glass[glass == 100])/(len(glass[glass == 255])+len(glass[glass == 100])):.2f}% Filled'
                else:
                    percentage = f'0% Filled'
            
                annotator.box_label(box, label=percentage)

        annotated_image = annotator.result()

        ret, buffer = cv2.imencode('.jpg', annotated_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')


        
    else:
        ret, buffer = cv2.imencode('.jpg', opencvImage)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

    payload = {
        "image": encoded_image,
        "metadata": {
            "description": "processed image"
        }
    }
    return jsonify(payload)

if __name__ == '__main__':
    app.run(port=9000)
