import base64
import io

import cv2
import numpy as np
import requests
import logging
import time
import glob
import os
from PIL import Image

# HOST = os.environ.get("HOST", "127.0.0.1")
# PORT = os.environ.get("PORT", 12000)
# threshold = 0.5
#
# fpath = glob.glob("./resources/samples/*.JPG")[0]
# response = requests.post(f"http://{HOST}:{PORT}/detection", files={"file": open(fpath, 'rb')})


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    HOST = os.environ.get("HOST", "127.0.0.1")
    PORT = os.environ.get("PORT", 12002)
    threshold = 0.5

    if not os.path.exists("response/samples"):
        os.makedirs("./response/samples", exist_ok=True)

    fpaths = glob.glob("./resources/samples/*.JPG")[:50]

    # Capture frame-by-frame
    for i, fpath in enumerate(fpaths):
        # break
        start = time.time()
        response = requests.post(f"http://{HOST}:{PORT}/detection", files={"file": open(fpath, 'rb')})
        end = time.time()
        elapsed = end - start
        logging.info(f"took {elapsed:.4f} sec")

        idx2fish = {0: "of", 1: "shrimp"}

        if not response.ok:
            logging.error(response.text)
        data = response.json()

        frame = cv2.imread(fpath)

        zipped = zip(
            data['fish_category'],
            data['bboxes'],
            data['diseases'],
            data['diseases_scores'],
            data['is_whole_shape'],
            data['anomaly_maps'],
        )

        for j, (fish_category, bbox, is_disease, prob, is_target, anomaly_map) in enumerate(zipped):
            xmin, ymin, xmax, ymax = bbox

            # 광어가 아닌 경우
            if fish_category != 0:
                color = (128, 128, 128)
                cv2.putText(frame, idx2fish[fish_category], (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 5)
                continue

            # 광어이지만 전체 몸통이 보이지는 않는 경우
            if not is_target:
                color = (243, 190, 203)
                cv2.putText(frame, "OF-occluded", (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 5)
                continue

            label = is_disease > threshold
            prob_str = ', '.join([ f'{i}:'+str(round(prob[i],3)) for i in prob])

            if label and anomaly_map:

                h, w = ymax - ymin, xmax - xmin

                max_size = max(w, h)
                pad_w = max_size - w
                pad_h = max_size - h

                pad_left = pad_w // 2
                pad_top = pad_h // 2
                pad_right = pad_w - pad_left
                pad_bottom = pad_h - pad_top

                anomaly_map = Image.open(io.BytesIO(base64.b64decode(anomaly_map)))
                anomaly_map = anomaly_map.resize((max_size, max_size))
                anomaly_map = np.array(anomaly_map)[pad_top:max_size-pad_bottom+1, pad_left:max_size-pad_right+1]
                anomaly_map = cv2.resize(anomaly_map, dsize=(w, h))

                mask = (anomaly_map > 125).any(axis=-1, keepdims=True) #125

                roi = frame[ymin:ymax, xmin:xmax, :]
                frame[ymin:ymax, xmin:xmax, :] = roi * ~mask + cv2.addWeighted(roi * mask, 0.8, anomaly_map * mask, 0.2, 0)

            color = (36, 255, 12) if not label else (36, 36, 255)
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 5)
            text = f"OF: Abnormal" if label else "OF: Normal"
            #prob = prob if label else 1 - prob
            cv2.putText(frame, f"{text}({prob_str})", (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

        cv2.imwrite(f"response/{os.path.basename(fpath)}", frame)
