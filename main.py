from Recognition import *
from settings import FACE_FOLD


video_path = FACE_FOLD.joinpath('farazDB.mp4')
image_path = FACE_FOLD.joinpath('faraz.jpeg')

image_src = cv2.imread(str(image_path))

cap = cv2.VideoCapture(str(video_path))
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_frames = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

n_frame = 0
ret = True

while n_frame < frameCount and ret:
    ret, video_frames[n_frame] = cap.read()
    n_frame += 1

print(np.shape(video_frames))

flag_liveness, flag_verification = face_recognition(video_frames, image_src)
print('Liveness:', flag_liveness, '\nVerification:', flag_verification)
