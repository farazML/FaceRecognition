from __future__ import print_function
from settings import (FACE_FOLD)
from Verificator import *
from settings import OX_RETINA_MODEL

import cv2
import datetime
from threading import Thread
import argparse
import imutils

video_path = FACE_FOLD.joinpath('farazb.mp4')



class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src=0): # src must replace with video_path
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-frames", type=int, default=20,
                    help="# of frames to loop over for FPS test")
    ap.add_argument("-d", "--display", type=int, default=1,
                    help="Whether or not frames should be displayed")
    args = vars(ap.parse_args())

    import onnxruntime as ort
    import onnxruntime.backend

    sess_options = ort.SessionOptions()
    print(OX_RETINA_MODEL)
    ort_sess = ort.InferenceSession(str(OX_RETINA_MODEL), sess_options, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

    print(ort.get_device())

    video, face, embed_src, embed_name_dict = find_image()
    flag = False

    # created a *threaded* video stream, allow the camera sensor to warmup,
    # and start the FPS counter
    print("[INFO] sampling THREADED frames from webcam...")
    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    # loop over some frames...this time using the threaded stream
    while fps._numFrames < args["num_frames"]:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        boxes, points = face_dm.detect(frame, max_num=1)

        bbox = []
        embed = []
        for idx in range(boxes.shape[0]):
            bbox = boxes[idx, 0:4]
            det_score = boxes[idx, 4]

            kps = None
            if points is not None:
                kps = points[idx]

            face = norm_crop(frame, kps)
            embed = arc_face.get_feat(face)

        sim = compute_sim(embed_src, embed)
        if sim > 0.5:
            flag = True

        # check to see if the frame should be displayed to our screen
        if args["display"] > 0:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
        # update the FPS counter
        fps.update()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print(flag)
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


