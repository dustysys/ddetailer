import os
import sys
import cv2
from PIL import Image
import numpy as np
import gradio as gr

from modules import processing, images
from modules import scripts, script_callbacks, shared, devices, modelloader
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
from modules.sd_models import model_hash
from modules.paths import models_path
from basicsr.utils.download_util import load_file_from_url

dd_models_path = os.path.join(models_path, "mmdet")

def list_models(model_path):
        model_list = modelloader.load_models(model_path=model_path, ext_filter=[".pth"])
        
        def modeltitle(path, shorthash):
            abspath = os.path.abspath(path)

            if abspath.startswith(model_path):
                name = abspath.replace(model_path, '')
            else:
                name = os.path.basename(path)

            if name.startswith("\\") or name.startswith("/"):
                name = name[1:]

            shortname = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]

            return f'{name} [{shorthash}]', shortname
        
        models = []
        for filename in model_list:
            h = model_hash(filename)
            title, short_model_name = modeltitle(filename, h)
            models.append(short_model_name + "[" + h + "]")
        
        return models

def startup():
    from launch import is_installed, run
    if not is_installed("mmdet"):
        python = sys.executable
        run(f'"{python}" -m pip install -U openmim', desc="Installing openmim", errdesc="Couldn't install openmim")
        run(f'"{python}" -m mim install mmcv-full', desc=f"Installing mmcv-full", errdesc=f"Couldn't install mmcv-full")
        run(f'"{python}" -m pip install mmdet', desc=f"Installing mmdet", errdesc=f"Couldn't install mmdet")

    if (len(list_models(dd_models_path)) == 0):
        print("No detection models found, downloading...")
        bbox_path = os.path.join(dd_models_path, "bbox")
        segm_path = os.path.join(dd_models_path, "segm")
        load_file_from_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/bbox/mmdet_anime-face_yolov3.pth", bbox_path)
        load_file_from_url("https://huggingface.co/dustysys/ddetailer/raw/main/mmdet/bbox/mmdet_anime-face_yolov3.py", bbox_path)
        load_file_from_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/segm/mmdet_dd-person_mask2former.pth", segm_path)
        load_file_from_url("https://huggingface.co/dustysys/ddetailer/raw/main/mmdet/segm/mmdet_dd-person_mask2former.py", segm_path)

startup()

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}

class DetectionDetailerScript(scripts.Script):
    def title(self):
        return "Detection Detailer"
    
    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        import modules.ui

        model_list = list_models(dd_models_path)
        model_list.insert(0, "None")
        if is_img2img:
            info = gr.HTML("<p style=\"margin-bottom:0.75em\">Recommended settings: Use from inpaint tab, inpaint at full res ON, denoise <0.5</p>")
        else:
            info = gr.HTML("")
        with gr.Group():
            with gr.Row():
                dd_model_a = gr.Dropdown(label="Primary detection model (A)", choices=model_list,value = "None", visible=True, type="value")
            
            with gr.Row():
                dd_conf_a = gr.Slider(label='Detection confidence threshold % (A)', minimum=0, maximum=100, step=1, value=80, visible=False)
                dd_dilation_factor_a = gr.Slider(label='Dilation factor (A)', minimum=0, maximum=255, step=1, value=1, visible=False)
            
            with gr.Row():
                dd_preprocess_b = gr.Checkbox(label='Inpaint model B detections before model A runs', value=False, visible=False)
                dd_bitwise_and_b = gr.Checkbox(label='Bitwise AND model A and B detections ', value=True, visible=False)  
        
        br = gr.HTML("<br>")

        with gr.Group():
            with gr.Row():
                dd_model_b = gr.Dropdown(label="Secondary detection model (B) (optional)", choices=model_list,value = "None", visible =False, type="value")

            with gr.Row():
                dd_conf_b = gr.Slider(label='Detection confidence threshold % (B)', minimum=0, maximum=100, step=1, value=80, visible=False)
                dd_dilation_factor_b = gr.Slider(label='Dilation factor (B)', minimum=0, maximum=255, step=1, value=1, visible=False)
        
        with gr.Group():
            with gr.Row():
                dd_mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, visible=(not is_img2img))
                dd_inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='original', type="index", visible=(not is_img2img))
            
            with gr.Row():
                dd_inpaint_full_res = gr.Checkbox(label='Inpaint at full resolution ', value=True, visible = (not is_img2img))
                dd_inpaint_full_res_padding = gr.Slider(label='Inpaint at full resolution padding, pixels', minimum=0, maximum=256, step=4, value=32, visible=(not is_img2img))

        dd_model_a.change(
            lambda modelname: {
                dd_model_b:gr_show( modelname != "None" ),
                dd_conf_a:gr_show( modelname != "None" ),
                dd_dilation_factor_a:gr_show( modelname != "None")

            },
            inputs= [dd_model_a],
            outputs =[dd_model_b, dd_conf_a, dd_dilation_factor_a]
        )

        dd_model_b.change(
            lambda modelname: {
                dd_preprocess_b:gr_show( modelname != "None" ),
                dd_bitwise_and_b:gr_show( modelname != "None" ),
                dd_conf_b:gr_show( modelname != "None" ),
                dd_dilation_factor_b:gr_show( modelname != "None")
            },
            inputs= [dd_model_b],
            outputs =[dd_preprocess_b, dd_bitwise_and_b, dd_conf_b, dd_dilation_factor_b]
        )
        
        return [info,
                dd_model_a, 
                dd_conf_a, dd_dilation_factor_a,
                dd_preprocess_b, dd_bitwise_and_b, 
                br,
                dd_model_b,
                dd_conf_b, dd_dilation_factor_b,  
                dd_mask_blur, dd_inpainting_fill,
                dd_inpaint_full_res, dd_inpaint_full_res_padding
        ]

    def run(self, p, info,
                     dd_model_a, 
                     dd_conf_a, dd_dilation_factor_a,
                     dd_preprocess_b, dd_bitwise_and_b, 
                     br,
                     dd_model_b,
                     dd_conf_b, dd_dilation_factor_b, 
                     dd_mask_blur, dd_inpainting_fill,
                     dd_inpaint_full_res, dd_inpaint_full_res_padding):

        processing.fix_seed(p)
        initial_info = None
        seed = p.seed
        init_image = p.init_images[0]
        devices.torch_gc()
        p.batch_size = 1
        ddetail_count = p.n_iter
        p.n_iter = 1
        p.mask_blur = dd_mask_blur
        p.inpainting_fill = dd_inpainting_fill
        p.inpaint_full_res = dd_inpaint_full_res
        p.inpaint_full_res_padding = dd_inpaint_full_res_padding
        p.do_not_save_grid = True
        p.do_not_save_samples = True
        masks_a = []
        masks_b_pre = []
        
        # Optional secondary pre-processing run
        if (dd_model_b != "None" and dd_preprocess_b):
            label_b_pre = "B"
            results_b_pre = inference(init_image, dd_model_b, dd_conf_b/100.0, label_b_pre)
            masks_b_pre = create_segmasks(results_b_pre)
            masks_b_pre = dilate_masks(masks_b_pre, dd_dilation_factor_b, 1)
            if (len(masks_b_pre) > 0):
                results_b_pre = update_result_masks(results_b_pre, masks_b_pre)
                shared.state.current_image = create_segmask_preview(results_b_pre, init_image)
                gen_count = len(masks_b_pre)
                state.job_count = ddetail_count * gen_count
                print(f"Processing {len(masks_b_pre)} model {label_b_pre} detections per output image for a total of {state.job_count} generation(s).")
                for n in range(ddetail_count):
                    start_seed = seed + n
                    p.seed = start_seed

                    for i in range(gen_count):
                        p.image_mask = masks_b_pre[i]
                        state.job = f"Generation {i + 1 + n} out of {state.job_count}"
                        processed = processing.process_images(p)
                        if initial_info is None:
                            initial_info = processed.info
                        p.seed = processed.seed + 1
                        p.init_images = processed.images

                if opts.samples_save:
                    images.save_image(processed.images[0], p.outpath_samples, "", start_seed, p.prompt, opts.samples_format, info=initial_info, p=p)
            else:
                print("No model B detections with current settings.")

        # Primary run
        if (dd_model_a != "None"):
            init_image = p.init_images[0]
            label_a = "A"
            if (dd_bitwise_and_b):
                label_a = "A AND B"
            results_a = inference(init_image, dd_model_a, dd_conf_a/100.0, label_a)
            masks_a = create_segmasks(results_a)
            masks_a = dilate_masks(masks_a, dd_dilation_factor_a, 1)
            if (dd_model_b != "None" and dd_bitwise_and_b):
                label_b = "B"
                results_b = inference(init_image, dd_model_b, dd_conf_b/100.0, label_b)
                masks_b = create_segmasks(results_b)
                masks_b = dilate_masks(masks_b, dd_dilation_factor_b, 1)
                if (len(masks_b) > 0):
                    combined_mask_b = combine_masks(masks_b)
                    for i in reversed(range(len(masks_a))):
                        masks_a[i] = bitwise_and_masks(masks_a[i], combined_mask_b)
                        masks_a[i].save("output" + str(i) + ".png", "PNG")
                        if (is_allblack(masks_a[i])):
                            del masks_a[i]
                            for result in results_a:
                                del result[i]
                                
                else:
                    print("No model B detections to overlap with model A masks")
                    results_a = []
                    masks_a = []
            
            if (len(masks_a) > 0):
                results_a = update_result_masks(results_a, masks_a)
                shared.state.current_image = create_segmask_preview(results_a, init_image)
                gen_count = len(masks_a)
                state.job_count = ddetail_count * gen_count
                print(f"Processing {len(masks_a)} model {label_a} detections per output image for a total of {state.job_count} generation(s).")
                for n in range(ddetail_count):
                    start_seed = seed + n
                    p.seed = start_seed

                    for i in range(gen_count):
                        p.image_mask = masks_a[i]
                        state.job = f"Generation {i + 1 + n} out of {state.job_count}"
                        processed = processing.process_images(p)
                        if initial_info is None:
                            initial_info = processed.info
                        p.seed = processed.seed + 1
                        p.init_images = processed.images
                        
                if opts.samples_save:
                    images.save_image(processed.images[0], p.outpath_samples, "", start_seed, p.prompt, opts.samples_format, info=initial_info, p=p)
            else: 
                print("No model A detections with current settings")
                
        if ( len(masks_a) == 0 and len(masks_b_pre) == 0):
            print("No detections to process")
            output_images = p.init_images
            initial_info = "No detections to process"
        else:
            output_images = processed.images
        
        return Processed(p, output_images, seed, initial_info)

def modeldataset(model_shortname):
    path = modelpath(model_shortname)
    if ("mmdet" in path and "segm" in path):
        dataset = 'coco'
    else:
        dataset = 'bbox'
    return dataset

def modelpath(model_shortname):
    model_list = modelloader.load_models(model_path=dd_models_path, ext_filter=[".pth"])
    model_h = model_shortname.split("[")[-1].split("]")[0]
    for path in model_list:
        if ( model_hash(path) == model_h):
            return path

def update_result_masks(results, masks):
    for i in range(len(masks)):
        boolmask = np.array(masks[i], dtype=bool)
        results[2][i] = boolmask
    return results

def create_segmask_preview(results, image):

    labels = results[0]
    bboxes = results[1]
    segms = results[2]

    cv2_image = np.array(image)
    cv2_image = cv2_image[:, :, ::-1].copy()

    for i in range(len(segms)):
        color = np.full_like(cv2_image, np.random.randint(100, 256, (1, 3), dtype=np.uint8))
        alpha = 0.2
        color_image = cv2.addWeighted(cv2_image, alpha, color, 1-alpha, 0)
        cv2_mask = segms[i].astype(np.uint8) * 255
        cv2_mask_bool = np.array(segms[i], dtype=bool)
        centroid = np.mean(np.argwhere(cv2_mask_bool),axis=0)
        centroid_x, centroid_y = int(centroid[1]), int(centroid[0])

        cv2_mask_rgb = cv2.merge((cv2_mask, cv2_mask, cv2_mask))
        cv2_image = np.where(cv2_mask_rgb == 255, color_image, cv2_image)
        text_color = tuple([int(x) for x in ( color[0][0] - 100 )])
        name = labels[i]
        score = bboxes[i][4]
        score = str(score)[:4]
        text = name + ":" + score
        cv2.putText(cv2_image, text, (centroid_x - 30, centroid_y), cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1, cv2.LINE_AA)
    
    if ( len(segms) > 0):
        preview_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    else:
        preview_image = image

    return preview_image

def is_allblack(mask):
    cv2_mask = np.array(mask)
    return cv2.countNonZero(cv2_mask) == 0

def bitwise_and_masks(mask1, mask2):
    cv2_mask1 = np.array(mask1)
    cv2_mask2 = np.array(mask2)
    cv2_mask = cv2.bitwise_and(cv2_mask1, cv2_mask2)
    mask = Image.fromarray(cv2_mask)
    return mask

def dilate_masks(masks, dilation_factor, iter=1):
    if dilation_factor == 0:
        return masks
    dilated_masks = []
    kernel = np.ones((dilation_factor,dilation_factor), np.uint8)
    for i in range(len(masks)):
        cv2_mask = np.array(masks[i])
        dilated_mask = cv2.dilate(cv2_mask, kernel, iter)
        dilated_masks.append(Image.fromarray(dilated_mask))
    return dilated_masks

def combine_masks(masks):
    initial_cv2_mask = np.array(masks[0])
    combined_cv2_mask = initial_cv2_mask
    for i in range(1, len(masks)):
        cv2_mask = np.array(masks[i])
        combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
    
    combined_mask = Image.fromarray(combined_cv2_mask)
    return combined_mask

def on_ui_settings():
    shared.opts.add_option("dd_same_seed", shared.OptionInfo(False, "Use same seed for all sub-images", section=("ddetailer", "Detection Detailer")))

def create_segmasks(results):
    segms = results[2]
    segmasks = []
    for i in range(len(segms)):
        cv2_mask = segms[i].astype(np.uint8) * 255
        mask = Image.fromarray(cv2_mask)
        segmasks.append(mask)

    return segmasks

import mmcv
from mmdet.core import get_classes
from mmdet.apis import (inference_detector,
                        init_detector)

def get_device():
    device_id = shared.cmd_opts.device_id
    if device_id is not None:
        cuda_device = f"cuda:{device_id}"
    else:
        cuda_device = "cpu"
    return cuda_device

def inference(image, modelname, conf_thres, label):
    path = modelpath(modelname)
    if ( "mmdet" in path and "bbox" in path ):
        results = inference_mmdet_bbox(image, modelname, conf_thres, label)
    elif ( "mmdet" in path and "segm" in path):
        results = inference_mmdet_segm(image, modelname, conf_thres, label)
    return results

def inference_mmdet_segm(image, modelname, conf_thres, label):
    model_checkpoint = modelpath(modelname)
    model_config = os.path.splitext(model_checkpoint)[0] + ".py"
    model_device = get_device()
    model = init_detector(model_config, model_checkpoint, device=model_device)
    mmdet_results = inference_detector(model, np.array(image))
    bbox_results, segm_results = mmdet_results
    dataset = modeldataset(modelname)
    classes = get_classes(dataset)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_results)
    ]

    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_results)
    segms = mmcv.concat_list(segm_results)
    filter_inds = np.where(bboxes[:,-1] > conf_thres)[0]
    results = [[],[],[]]
    for i in filter_inds:
        results[0].append(label + "-" + classes[labels[i]])
        results[1].append(bboxes[i])
        results[2].append(segms[i])

    return results

def inference_mmdet_bbox(image, modelname, conf_thres, label):
    model_checkpoint = modelpath(modelname)
    model_config = os.path.splitext(model_checkpoint)[0] + ".py"
    model_device = get_device()
    model = init_detector(model_config, model_checkpoint, device=model_device)
    results = inference_detector(model, np.array(image))
    cv2_image = np.array(image)
    cv2_image = cv2_image[:, :, ::-1].copy()
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for (x0, y0, x1, y1, conf) in results[0]:
        cv2_mask = np.zeros((cv2_gray.shape), np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    bboxes = np.vstack(results[0])
    filter_inds = np.where(bboxes[:,-1] > conf_thres)[0]
    results = [[],[],[]]
    for i in filter_inds:
        results[0].append(label)
        results[1].append(bboxes[i])
        results[2].append(segms[i])

    return results

script_callbacks.on_ui_settings(on_ui_settings)