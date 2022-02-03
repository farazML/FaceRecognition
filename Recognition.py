from Detection import *
from LivenessDetection import *
import cv2


def face_recognition(video, image):
    try:
        detection = Recognition()
        face_src, embed_src, bbox_src, boxes_src, kps_src, det_score_src = detection.detection(image)

    except:
        raise Exception('something went wrong in detection module.')

    # these variables related to EyeBlinkDetector, Creating a list eye_blink_signal
    eye_blink_signal = []

    # Creating an object blink_ counter
    blink_counter = 0
    previous_ratio = 100
    flag_verification = False
    flag_liveness = False

    # this is for store similarity between source and detected embed vector
    similarity = []

    if len(video) == 0:
        raise Exception('video file is empty! check this.')

    for frame in video:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detection and Calculate embed vector
        face, embed, bbox, boxes, kps, det_score = detection.detection(frame)

        # dlib predictor works with own bounding box format, so we do this
        x, y, x1, y1 = bbox.astype(np.int32)
        face_bbox = dlib.rectangle(x, y, x1, y1)

        # START Verification Based on Similarity between source(embed_src) and Detected(embed) embed vector
        sim = detection.verification(embed_src, embed)
        if float(sim) > 0.5:
            flag_verification = True

            similarity.append(sim)
            # START Liveness Detection Using EyeBlinkDetector
            # Creating an obj in which we will store detected facial landmarks which calc by dlib predictor
            landmarks = predictor(gray, face_bbox)

            # Calculating left eye aspect ratio
            left_eye_ratio = get_EAR([36, 37, 38, 39, 40, 41], landmarks)

            # Calculating right eye aspect ratio
            right_eye_ratio = get_EAR([42, 43, 44, 45, 46, 47], landmarks)

            # Calculating aspect ratio for both eyes
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

            # Rounding blinking_ratio on two decimal places
            blinking_ratio_1 = blinking_ratio * 100
            blinking_ratio_2 = np.round(blinking_ratio_1)
            blinking_ratio_rounded = blinking_ratio_2 / 100

            # Appending blinking ratio to a list eye_blink_signal
            eye_blink_signal.append(blinking_ratio)

            # TODO: blinking_ratio can be tune
            if blinking_ratio < 0.20:
                if previous_ratio > 0.20:
                    blink_counter = blink_counter + 1

            previous_ratio = blinking_ratio
            # END Liveness Detection

            if blink_counter >= 5:
                flag_liveness = True
                if flag_liveness:
                    break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    return flag_liveness, flag_verification
