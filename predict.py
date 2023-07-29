from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2


# Load config from a config file
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = './output/model_0002999.pth'
cfg.MODEL.DEVICE = 'cpu'

# Create predictor instance
predictor = DefaultPredictor(cfg)

# Load image
image = cv2.imread("./data/sample_jpg/EC1-L1-S70.jpg")

# Perform prediction
outputs = predictor(image)
print(outputs)
#threshold = 0.5

# Display predictions
preds = outputs["instances"].pred_classes.tolist()
scores = outputs["instances"].scores.tolist()
bboxes = outputs["instances"].pred_boxes
print(scores[0],"*************")
threshold = scores[0]

for j, bbox in enumerate(bboxes):
    bbox = bbox.tolist()

    score = scores[j]
    pred = preds[j]

    if score >= threshold and pred == 0:
        x1, y1, x2, y2 = [int(i) for i in bbox]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)

cv2.imshow('image', image)
cv2.waitKey(0)
