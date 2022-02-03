import tensorflow as tf
import onnxruntime as ort
import os
from tr.networks.retina.model import RetinaFace
from tr.networks.arc_face.models import ArcFaceONNX
from tr.utils import Face, norm_crop
from settings import (OX_RETINA_MODEL,
                      OX_ARC_FACE_PATH,
                      RETINA_CONF,
                      RETINA_NMS_CONF,
                      FACE_FOLD,
                      RESIZE_SHAPE)

# this is about log file
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ort.set_default_logger_severity(3)
print(ort.get_device())

# this is for memory growth management
physical_devices = tf.config.list_physical_devices('GPU')


class Recognition:
    def __init__(self):
        self.face_dm = RetinaFace(model_file=str(OX_RETINA_MODEL), nms_thresh=RETINA_NMS_CONF, det_thresh=RETINA_CONF)
        self.face_dm.prepare(ctx_id=0, input_size=(RESIZE_SHAPE[1], RESIZE_SHAPE[0]))

        self.arc_face = ArcFaceONNX(model_file=str(OX_ARC_FACE_PATH))
        self.arc_face.prepare(ctx_id=0)

    def detection(self, image):
        boxes, points = self.face_dm.detect(image, max_num=1)

        for idx in range(boxes.shape[0]):
            bbox = boxes[idx, 0:4]
            det_score = boxes[idx, 4]

            kps = None
            if points is not None:
                kps = points[idx]

            face = norm_crop(image, kps)
            embed = self.arc_face.get_feat(face)

            return face, embed, bbox, boxes, kps, det_score

    def verification(self, emb1, emb2):
        similarity = self.arc_face.compute_sim(emb1, emb2)

        return similarity



