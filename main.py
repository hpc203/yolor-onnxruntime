import onnxruntime as ort
import cv2
import numpy as np
import argparse

class yolor():
    def __init__(self, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        self.classes = list(map(lambda x: x.strip(), open('label.txt', 'r').readlines()))
        self.inpWidth = 640
        self.inpHeight = 640
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession('yolor-p6-640-640.onnx', so)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
        self.keep_ratio = True
        self.swaprgb = True
    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)
    def resize_image(self, srcimg):
        top, left, newh, neww = 0, 0, self.inpHeight, self.inpWidth
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.inpWidth - neww - left, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.inpHeight - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left
    def detect(self, frame):
        srcimg = frame.copy()
        if self.swaprgb:
            srcimg = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        img, newh, neww, top, left = self.resize_image(srcimg)
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32) / 255.0
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

        outs = outs[outs[:,4] > self.objThreshold]
        srcimgHeight = srcimg.shape[0]
        srcimgWidth = srcimg.shape[1]
        ratioh, ratiow = srcimgHeight / newh, srcimgWidth / neww
        boxes = outs[:, :4]
        boxes[:, 0] = (boxes[:, 0] - 0.5 * boxes[:, 2] - left) * ratiow
        boxes[:, 1] = (boxes[:, 1] - 0.5 * boxes[:, 3] - top) * ratioh
        boxes[:, 2] = boxes[:, 2] * ratiow
        boxes[:, 3] = boxes[:, 3] * ratioh
        boxes = boxes.astype(np.int)
        classIds = np.argmax(outs[:,5:], axis = 1)
        confidences = np.max(outs[:,5:], axis = 1)

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), self.confThreshold, self.nmsThreshold)
        results = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            results.append((box[0], box[1], box[0] + box[2], box[1] + box[3], self.classes[classIds[i]]))
            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), thickness=1)
            cv2.putText(frame, self.classes[classIds[i]]+': '+str(round(confidences[i], 3)), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
        return frame

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='bus.jpg', help="image path")
    parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    parser.add_argument('--objThreshold', default=0.5, type=float, help='object confidence')
    args = parser.parse_args()

    net = yolor(confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold, objThreshold=args.objThreshold)
    srcimg = cv2.imread(args.imgpath)
    srcimg = net.detect(srcimg)

    winName = 'Deep learning object detection in ONNXRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()