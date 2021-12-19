# Copyright (c) OpenMMLab. All rights reserved.
# Modified from thirdparty/mmdetection/demo/image_demo.py
import asyncio
import glob
import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from argparse import ArgumentParser

from mmcv import Config
from mmdet.apis import async_inference_detector, inference_detector, show_result_pyplot

from ssod.apis.inference import init_detector, save_result
from ssod.utils import patch_config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("img", help="Image file")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--score-thr", type=float, default=0.5, help="bbox score threshold"
    )
    parser.add_argument(
        "--async-test",
        action="store_true",
        help="whether to set async options for async inference.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="specify the directory to save visualization results.",
    )
    args = parser.parse_args()
    return args


def main(args):
    cfg = Config.fromfile(args.config)
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, args.checkpoint, device=args.device)
    imgs = glob.glob(args.img)
    for img in imgs:
        # test a single image
        result = inference_detector(model, img)
        # show the results
        if args.output is None:
            show_result_pyplot(model, img, result, score_thr=args.score_thr)
        else:
            out_file_path = os.path.join(args.output, os.path.basename(img))
            print(f"Save results to {out_file_path}")
            save_result(
                model, img, [result[0]], score_thr=args.score_thr, out_file=out_file_path
            )
            blur_img(img, result[0])
        


async def async_main(args):
    cfg = Config.fromfile(args.config)
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, args.checkpoint, device=args.device)
    # test a single image
    args.img = glob.glob(args.img)
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    for img, pred in zip(args.img, result):
        if args.output is None:
            show_result_pyplot(model, img, pred, score_thr=args.score_thr)
        else:
            out_file_path = os.path.join(args.output, os.path.basename(img))
            print(f"Save results to {out_file_path}")
            save_result(
                model, img, pred, score_thr=args.score_thr, out_file=out_file_path
            )


def blur_img(img_name, detection, factor=3):
    img = cv2.imread(img_name)
    (h, w) = img.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)
    # ensure the width of the kernel is odd
    if kW % 2 == 0:
        kW -= 1
    # ensure the height of the kernel is odd
    if kH % 2 == 0:
        kH -= 1
        
    blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
    mask = np.zeros(img.shape, dtype=np.uint8)

    out = img
    for section in detection:
        score = section[-1]
        x, y, w, h, _ = section.astype(np.int)
        if score > 0.7:
            masked = cv2.rectangle(mask, (x, y), (w, h), (255, 255, 255), -1)
            out = np.where(masked!=np.array([255, 255, 255]), img, blurred_img)

    cv2.imwrite('output_blur/'+ img_name.split('/')[-1], out)


def blur_img_retina(img_name, faces, factor=3):
    img = cv2.imread(img_name)
    (h, w) = img.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)
    # ensure the width of the kernel is odd
    if kW % 2 == 0:
        kW -= 1
    # ensure the height of the kernel is odd
    if kH % 2 == 0:
        kH -= 1
        
    blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
    mask = np.zeros(img.shape, dtype=np.uint8)

    out = img

    if faces is not None:
        for face in faces:
            box = face['bbox'].astype(np.int)
            masked = cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)
            out = np.where(masked!=np.array([255, 255, 255]), img, blurred_img)

    cv2.imwrite('retinaface/output_blur/'+ img_name.split('/')[-1], out)



def predit_retina_face(args):

    """
    Measure times and points of list predictions
    
    ctx_id is for computacion on GPU minimun 8GB of vRAM
    
    """
    filename = 'retinaface/output/'


    imgs = glob.glob(args.img)
    for img in imgs:
        # test a single image
        image = cv2.imread(img)

        app = FaceAnalysis()
        app.prepare(ctx_id=0, det_size=(640, 640))
        faces = app.get(image)

        # show the results

        if args.output is None:
            print('------------------------')

        else:
            out_file_path = os.path.join(args.output, os.path.basename(img))
            print(f"Save results to {out_file_path}")

            rimg = app.draw_on(image, faces)
            cv2.imwrite(filename + img.split('/')[-1], rimg)

            blur_img_retina(img, faces, factor=3)






if __name__ == "__main__":
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        #main(args)
        predit_retina_face(args)


