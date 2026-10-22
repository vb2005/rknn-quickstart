import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite

MODEL_PATH = 'yolo26s-rk3588.rknn'
IMAGE_PATH = 'test.jpg'

INPUT_W = 640
INPUT_H = 640


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]

    r = min(new_shape[0] / h, new_shape[1] / w)

    new_w = int(round(w * r))
    new_h = int(round(h * r))

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    dw = new_shape[1] - new_w
    dh = new_shape[0] - new_h

    left = dw // 2
    right = dw - left

    top = dh // 2
    bottom = dh - top

    result = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return result, r, (left, top)


print('Create RKNNLite...')

rknn = RKNNLite()


print('Load model...')

ret = rknn.load_rknn(MODEL_PATH)

if ret != 0:
    print('ERROR: load_rknn failed:', ret)
    exit(ret)

print('Model loaded.')


print('Init RK3588 NPU...')

ret = rknn.init_runtime(
    core_mask=RKNNLite.NPU_CORE_0_1_2
)

if ret != 0:
    print('ERROR: init_runtime failed:', ret)
    exit(ret)

print('NPU initialized.')


print('Load image...')

img0 = cv2.imread(IMAGE_PATH)

if img0 is None:
    print('ERROR: cannot load:', IMAGE_PATH)
    rknn.release()
    exit(1)

print('Original image:', img0.shape)


img, scale, pad = letterbox(
    img0,
    (INPUT_H, INPUT_W)
)

#
# OpenCV -> BGR
# YOLO/RKNN обычно ожидает RGB
#
img = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)

#
# RKNN-модели обычно принимают NHWC uint8.
#
input_data = np.expand_dims(
    img,
    axis=0
)

input_data = np.ascontiguousarray(
    input_data,
    dtype=np.uint8
)

print('Input shape:', input_data.shape)
print('Input dtype:', input_data.dtype)

print()
print('Warm-up...')

for i in range(5):
    outputs = rknn.inference(
        inputs=[input_data]
    )

print('Warm-up done.')


print()
print('Inference...')

COUNT = 20

outputs = rknn.inference(inputs=[input_data])

start = time.perf_counter()

for i in range(COUNT):
    outputs = rknn.inference(
        inputs=[input_data]
    )

end = time.perf_counter()

elapsed = end - start

print()
print('===========================')
print('RKNN inference test')
print('===========================')

print('Iterations:', COUNT)
print('Total time: %.3f s' % elapsed)
print('Average: %.3f ms' % (
    elapsed / COUNT * 1000
))
print('FPS: %.2f' % (
    COUNT / elapsed
))


print()
print('Outputs:')
print('Number of outputs:', len(outputs))

for i, out in enumerate(outputs):

    print()
    print('Output', i)
    print(' shape:', out.shape)
    print(' dtype:', out.dtype)

    try:
        print(' min:', np.min(out))
        print(' max:', np.max(out))
        print(' mean:', np.mean(out))
    except:
        pass

    flat = out.reshape(-1)

    print(
        ' first values:',
        flat[:20]
    )


rknn.release()

print()
print('Done.')