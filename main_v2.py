import io
import os
from typing import List

import cv2
import numpy as np
import torch
import yaml
from easydict import EasyDict
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

import uvicorn
import matplotlib.pyplot as plt
import cv2

from src.utils import non_max_suppression, scale_coords, filter_invalid_bboxes, letterbox
from src.utils import IMAGENET_MEAN, IMAGENET_STD
from src.utils import sigmoid, encode_base64, postprocess_anomaly_maps
from src.inference_session import initialize_session, load_mask_kpt_rcnn
from src.patch_core import STPM
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def to_numpy(tensor):
    tensor = torch.Tensor(tensor)
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def paddingresize(obj_img, head_kpt, crop_size):
    obj_img = obj_img.astype(float)
    obj_img = np.pad(obj_img,
                     ((64, 64), (64, 64), (0, 0)),
                     'constant', constant_values=0)

    head_kpt[0] = head_kpt[0] + 64
    head_kpt[1] = head_kpt[1] + 64

    h, w, c = obj_img.shape
    if h >= w:
        ratio = crop_size / h
        obj_img = cv2.resize(obj_img, dsize=(int(w * ratio), crop_size))
        _, w, _ = obj_img.shape
        pad_t = crop_size - w
        pad_l = pad_t // 2
        obj_img = np.pad(obj_img,
                         ((0, 0), (pad_l, pad_t - pad_l), (0, 0)),
                         'constant', constant_values=0)
        # # keypoint
        # _kp_r[0][0] = int(_kp_r[0][0] * ratio) + pad_l
        # _kp_r[0][1] = int(_kp_r[0][1] * ratio)
        # _kp_r[1][0] = int(_kp_r[1][0] * ratio) + pad_l
        # _kp_r[1][1] = int(_kp_r[1][1] * ratio)
        head_kpt[0] = min(int(head_kpt[0] * ratio) + pad_l, 512)
        head_kpt[1] = min(int(head_kpt[1] * ratio), 512)

    elif h < w:
        ratio = crop_size / w
        obj_img = cv2.resize(obj_img, dsize=(crop_size, int(h * ratio)))
        h, _, _ = obj_img.shape
        pad_t = crop_size - h
        pad_l = pad_t // 2
        obj_img = np.pad(obj_img,
                         ((pad_l, pad_t - pad_l), (0, 0), (0, 0)),
                         'constant', constant_values=0)
        head_kpt[0] = min(int(head_kpt[0] * ratio), 512)
        head_kpt[1] = min(int(head_kpt[1] * ratio) + pad_l, 512)

    return obj_img, head_kpt

# 설정파일 로드
#MODEL_CONFIG_FPATH = os.environ.get("MODEL_CONFIG_FPATH")
MODEL_CONFIG_FPATH = './config/gmission.proto_v2.yml'                                                                      ###### 수정필요 대치
config = yaml.safe_load(open(MODEL_CONFIG_FPATH).read())
config = EasyDict(config)

# 모델 초기화
# detection & segmentation
DETECTION_SESSION = load_mask_kpt_rcnn(config.detection_kptmaskrcnn.model_path)
# DETECTION_SESSION.to(device)

FISH_CLASSIFICATION_SESSION = initialize_session(config.fish_classification.model_path,
                                                 config.fish_classification.device_id)
FISH_SHAPE_CLASSIFICATION_SESSION = initialize_session(config.fish_shape_classification.model_path,
                                                 config.fish_shape_classification.device_id)

# fish disease detection
DISEASE_DETECTION_OF_O2_SESSION = initialize_session(config.disease_of_o2.model_path,
                                                     config.disease_of_o2.device_id)
DISEASE_DETECTION_OF_SCRATCH_SESSION = initialize_session(config.disease_of_scratch.model_path,
                                                          config.disease_of_scratch.device_id)
DISEASE_DETECTION_OF_ULCER_SESSION = initialize_session(config.disease_of_ulcer.model_path,
                                                        config.disease_of_ulcer.device_id)
DISEASE_DETECTION_OF_MELANISM_SESSION = initialize_session(config.disease_of_melanism.model_path,
                                                           config.disease_of_melanism.device_id)
DISEASE_DETECTION_MODEL={'o2': DISEASE_DETECTION_OF_O2_SESSION,
                         'scratch':DISEASE_DETECTION_OF_SCRATCH_SESSION,
                         'ulcer':DISEASE_DETECTION_OF_ULCER_SESSION,
                         'melanism':DISEASE_DETECTION_OF_MELANISM_SESSION}


# fish anomaly detection
PATCHCORE_SESSION = STPM(
    config.patchcore.repo_name,
    config.patchcore.model_name,
    config.patchcore.index_path,
    config.patchcore.k,
    config.patchcore.threshold,
    config.patchcore.device_id,
)

app = FastAPI()


@torch.no_grad()
@app.post("/detection")
async def get_detection_and_diseases(file: UploadFile) -> JSONResponse:
    """
    업로드된 이미지에 대해 넙치 객체의 바운딩 박스, 질병 여부를 반환
    - bboxes: x1, y1, x2, y2 순서 (Top left & bottom right)
    - diseases: 개체에 하나 이상의 질병이 있는 경우 1, 없는 경우 0
    :param file: 업로드 파일
    :return: JSONResponse
    """

    # if not file.content_type.startswith("image"):
    #     raise HTTPException(400, detail=f"Invalid content type: {file.content_type}")

    contents = await file.read()
    img_stream = io.BytesIO(contents)
    img0 = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)

    ####################################################################################################################
    # 1. Detection & Segmentation
    ####################################################################################################################

    # org_shape = img0.shape[:-1]

    # 모델 인풋 사이즈에 맞춰서 리사이징 (가로:세로 비율 보존)
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    # img = letterbox(img, (config.detection.input_size, config.detection.input_size), auto=False)
    img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
    img = torch.from_numpy(img/255.)
    output = DETECTION_SESSION(img)
    # ['boxes', 'labels', 'scores', 'masks', 'keypoints', 'keypoints_scores']
    if len(output) == 1:
        output = output[0]
    else:
        assert('Warning : cannot preocess detection outputs with batch images')

    obj_labels = output['labels'].detach() # 1 of, 2 shrimp
    conf_scores = output['scores'].detach()
    bboxes = output['boxes'].round().type(torch.int).detach()
    segmentations = torch.squeeze(output['masks']).detach()
    keypoints = output['keypoints'].detach()

    mask = conf_scores.numpy() > config.detection_kptmaskrcnn.conf_thres
    obj_labels = obj_labels[mask]
    conf_scores = conf_scores[mask]
    bboxes = bboxes[mask]
    segmentations = segmentations[mask]
    keypoints = keypoints[mask]
    head_keypoint = keypoints[:,1,:]


    bboxes, conf_scores = filter_invalid_bboxes(bboxes, conf_scores)
    segmentations = (segmentations > config.detection_kptmaskrcnn.seg_thres).to(int).numpy()

    num_objects = len(bboxes)

    if not num_objects:
        content = dict(
            bboxes=[],
            conf_scores=[],
            diseases=[],
            diseases_scores=[],
            anomaly_maps=[],
            whole_shape=[],
            fish_category=[],
            num_objects=num_objects
        )
        return JSONResponse(content=content)

    diseases_scores = np.array(["" for _ in range(num_objects)], dtype=object)
    anomaly_maps = np.array(["" for _ in range(num_objects)], dtype=object)
    diseases = np.array([0 for _ in range(num_objects)], dtype=np.uint8)
    is_whole_shape = np.array([0 for _ in range(num_objects)], dtype=np.uint8)
    mask = np.array([False for _ in range(num_objects)], dtype=np.bool_)

    # extract rois from the original input while preserving aspects

    img_rois: List[np.ndarray] = []

    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)  # RGB order
    for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
        roi = img[ymin:ymax + 1, xmin:xmax + 1, :]
        segm = segmentations[i][ymin:ymax + 1, xmin:xmax + 1]
        segm = np.expand_dims(segm, -1)
        fused_img = roi * segm
        img_rois.append(fused_img/255.)


        head_x, head_y, _ = head_keypoint[i]
        head_keypoint[i][0] = max(head_x - xmin, 0)
        head_keypoint[i][1] = max(head_y - ymin, 0)

    # padding
    crop_size = 512
    head_kpt_rois: List[np.ndarray] = []
    for i, obj_img in enumerate(img_rois):
        img_rois[i], h_kpt = paddingresize(obj_img, head_keypoint[i], crop_size)
        h_kpt = h_kpt.numpy().astype(int)
        img_k_rois = np.pad(img_rois[i], ((50, 50), (50, 50), (0, 0)), 'constant', constant_values=0)
        img_k_rois = img_k_rois[h_kpt[1]:h_kpt[1]+100, h_kpt[0]:h_kpt[0]+100, :]
        head_kpt_rois.append(img_k_rois)


    ####################################################################################################################
    # 2. Fish category classification
    ####################################################################################################################
    dsize = (config.fish_classification.input_size, config.fish_classification.input_size)
    classification_rois = np.stack([cv2.resize(roi, dsize=dsize) for roi in img_rois])
    classification_rois = classification_rois.transpose(0, 3, 1, 2)  # B x C x H x W

    ort_inputs = {FISH_CLASSIFICATION_SESSION.get_inputs()[0].name: classification_rois.astype(np.float32)}
    fish_category = FISH_CLASSIFICATION_SESSION.run(None, ort_inputs)[0].argmax(axis=1)
    fish_category_mask = (fish_category == config.fish_classification.target_index)
    mask[:] = fish_category_mask

    if not fish_category_mask.any():
        content = dict(
            bboxes=bboxes.tolist(),
            conf_scores=conf_scores.tolist(),
            diseases=[],
            is_whole_shape=is_whole_shape.tolist(),
            diseases_scores=diseases_scores.tolist(),
            anomaly_maps=anomaly_maps.tolist(),
            num_objects=num_objects,
            fish_category=fish_category.tolist()
        )
        return JSONResponse(content=content)


    ####################################################################################################################
    # 3. Classification: 전체 형상 인지를 분류
    ####################################################################################################################
    dsize = (config.fish_classification.input_size, config.fish_classification.input_size)
    classification_rois = np.stack([cv2.resize(roi, dsize=dsize) for roi in img_rois])
    classification_rois = classification_rois.transpose(0, 3, 1, 2)  # B x C x H x W
    ort_inputs = {FISH_SHAPE_CLASSIFICATION_SESSION.get_inputs()[0].name: classification_rois.astype(np.float32)}
    logits = FISH_SHAPE_CLASSIFICATION_SESSION.run(None, ort_inputs)[0].flatten()
    # logits /= config.fish_shape_classification.temperature
    probs = sigmoid(logits)
    whole_shape_mask = (probs > config.fish_shape_classification.threshold)  # sigmoid & threshold 적용하도록 수정
    mask[fish_category_mask] = whole_shape_mask[fish_category_mask]
    # is_whole_shape[mask] = 1

    ####################################################################################################################
    # 4. Disease Detection & Anomaly detection
    ####################################################################################################################

    if whole_shape_mask.any():
        dsize = (config.fish_classification.input_size, config.fish_classification.input_size)
        anomaly_rois = np.stack([cv2.resize(roi, dsize=dsize) for roi in img_rois])
        anomaly_rois = anomaly_rois.transpose(0, 3, 1, 2)  # B x C x H x W
        # anomaly_rois = anomaly_rois # [whole_shape_mask]
        anomaly_kpt_rois = np.stack([cv2.resize(roi, dsize=dsize) for roi in head_kpt_rois])
        anomaly_kpt_rois = anomaly_kpt_rois.transpose(0, 3, 1, 2)  # B x C x H x W

        # 각 어종별 질병분류
        if config.fish_classification.target_index == 0:
            _score_dict = {'o2': None, 'scratch': None, 'ulcer': None, 'melanism': None}

            for _disease in DISEASE_DETECTION_MODEL:
                if _disease == 'o2':
                    ort_inputs = {DISEASE_DETECTION_MODEL[_disease].get_inputs()[0].name: anomaly_kpt_rois.astype(np.float32)}
                    scores = DISEASE_DETECTION_MODEL[_disease].run(None, ort_inputs)[0]
                    _score_dict[_disease] = sigmoid(scores.squeeze()).round(4).tolist()
                else:
                    ort_inputs = {DISEASE_DETECTION_MODEL[_disease].get_inputs()[0].name: anomaly_rois.astype(np.float32)}
                    scores = DISEASE_DETECTION_MODEL[_disease].run(None, ort_inputs)[0]
                    _score_dict[_disease] = sigmoid(scores.squeeze()).round(4).tolist()


            _disease_scores = []
            len_obj = len(whole_shape_mask)
            for i in range(len_obj):
                obj_disease_scores = {}
                for j in ['o2', 'scratch', 'ulcer', 'melanism']:
                    obj_disease_scores[j] = _score_dict[j][i]
                _disease_scores.append(obj_disease_scores)

            #[_disease_scores[0][i] > config.disease.threshold for i in _disease_scores[0]]

            is_disease = []
            for i in range(len_obj):
                is_disease.append(True in [_disease_scores[i][j] > config.disease.threshold for j in _disease_scores[i]])

            # heatmap = whole_shape & disease
            #is_whole_and_disease = np.logical_and(whole_shape_mask, is_disease)
            dsize = (config.fish_classification.input_size, config.fish_classification.input_size)
            _img_rois = np.stack([cv2.resize(roi, dsize=dsize) for roi in img_rois])
            anomaly_rois_sub = anomaly_rois[whole_shape_mask]
            _, _heatmaps = PATCHCORE_SESSION.batch_run(anomaly_rois_sub.astype(np.float32), config.patchcore.batch_size)

        # heatmap = whole_shape & disease
        # others = whole_shape
        heatmaps = postprocess_anomaly_maps(_heatmaps, dsize=dsize)
        heatmaps *= (_img_rois[whole_shape_mask].transpose(0, 3, 1, 2) > 0)
        anomaly_maps[whole_shape_mask] = [encode_base64(heatmap) for heatmap in heatmaps.transpose(0, 2, 3, 1)]
        diseases_scores[mask] = np.array(_disease_scores)[mask]
        diseases[mask] = np.array(is_disease)[mask].astype(np.uint8)
        is_whole_shape[mask] = 1

    # 응답 반환
    content = dict(
        bboxes=bboxes.tolist(),
        conf_scores=conf_scores.tolist(),
        diseases=diseases.tolist(),
        diseases_scores=diseases_scores.tolist(),
        is_whole_shape=is_whole_shape.astype(np.uint8).tolist(),
        anomaly_maps=anomaly_maps.tolist(),
        num_objects=len(bboxes),
        fish_category=fish_category.tolist()
    )

    return JSONResponse(content=content)

if __name__ == "__main__":                                                                                              ###### 수정필요 삭제
    uvicorn.run(app, host='127.0.0.1', port=12002, log_level='info')