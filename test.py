import torch
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / 'yolov5'))

from models.experimental import attempt_load

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = 'yolov5/runs/train/exp/weights/best.pt' 
model = attempt_load(model_path, device=device)  
model.eval() 

video_path = 'cropandweed.mp4'  
cap = cv2.VideoCapture(video_path)

crop_count = 0
weed_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))  
    img = np.array(img)
    img = img.astype(np.float32) / 255.0 
    img = torch.from_numpy(img).to(device)  
    img = img.permute(2, 0, 1).unsqueeze(0)  

    with torch.no_grad():  
        pred = model(img)[0]  

    for det in pred:  
        if det is not None and len(det):
            det[:, :4] = det[:, :4].clip(0, img.shape[2]) 

            for *xyxy, conf, cls in det:
                cls = cls.item()  
                label = f'{model.names[int(cls)]} {conf:.2f}'  
                
                if model.names[int(cls)] == 'crop':
                    crop_count += 1 
                elif model.names[int(cls)] == 'weed':
                    weed_count += 1  

                frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                frame = cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    cv2.putText(frame, f'Crops: {crop_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Weeds: {weed_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
