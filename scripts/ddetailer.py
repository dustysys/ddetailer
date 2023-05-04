import os
import sys
import cv2
from PIL import Image
import numpy as np
import gradio as gr
import shutil

from modules import processing, images
from modules import scripts, script_callbacks, shared, devices, modelloader
from modules.processing import Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules.shared import opts, cmd_opts, state
from modules.sd_models import model_hash
from modules.paths import models_path, script_path
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
            models.append(title)
        
        return models

def startup():
    from launch import is_installed, run
    import torch
    legacy = torch.__version__.split(".")[0] < "2"
    if not is_installed("mmdet"):
        python = sys.executable
        run(f'"{python}" -m pip install -U openmim', desc="Installing openmim", errdesc="Couldn't install openmim")
        if legacy:
            run(f'"{python}" -m mim install mmcv-full', desc=f"Installing mmcv-full", errdesc=f"Couldn't install mmcv-full")
            run(f'"{python}" -m pip install mmdet==2.28.2', desc=f"Installing mmdet", errdesc=f"Couldn't install mmdet")
        else:
            run(f'"{python}" -m mim install mmcv>==2.0.0', desc=f"Installing mmcv", errdesc=f"Couldn't install mmcv")
            run(f'"{python}" -m pip install mmdet>=3', desc=f"Installing mmdet", errdesc=f"Couldn't install mmdet")

    bbox_path = os.path.join(dd_models_path, "bbox")
    segm_path = os.path.join(dd_models_path, "segm")
    if (len(list_models(dd_models_path)) == 0):
        print("No detection models found, downloading...")
        load_file_from_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/bbox/mmdet_anime-face_yolov3.pth", bbox_path)
        if legacy:
            load_file_from_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/segm/mmdet_dd-person_mask2former.pth", segm_path)
        else:
            load_file_from_url(
                "https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_r50_8xb2-lsj-50e_coco/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth",
                segm_path,
                file_name="mmdet_dd-person_mask2former.pth")

    print("Check config files...")
    config_dir = os.path.join(script_path, "extensions", "ddetailer", "config")
    if legacy:
        configs = [ "mmdet_anime-face_yolov3.py", "mmdet_dd-person_mask2former.py" ]
    else:
        configs = [ "mmdet_anime-face_yolov3-v3.py", "mmdet_dd-person_mask2former-v3.py", "mask2former_r50_8xb2-lsj-50e_coco-panoptic.py", "coco_panoptic.py" ]

    destdir = bbox_path
    for confpy in configs:
        conf = os.path.join(config_dir, confpy)
        if not legacy:
            confpy = confpy.replace("-v3.py", ".py")
        dest = os.path.join(destdir, confpy)
        if not os.path.exists(dest):
            print(f"Copy config file: {confpy}..")
            shutil.copy(conf, dest)
        destdir = segm_path

    print("Done")

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
                dd_conf_a = gr.Slider(label='Detection confidence threshold % (A)', minimum=0, maximum=100, step=1, value=30, visible=False)
                dd_dilation_factor_a = gr.Slider(label='Dilation factor (A)', minimum=0, maximum=255, step=1, value=4, visible=False)

            with gr.Row():
                dd_offset_x_a = gr.Slider(label='X offset (A)', minimum=-200, maximum=200, step=1, value=0, visible=False)
                dd_offset_y_a = gr.Slider(label='Y offset (A)', minimum=-200, maximum=200, step=1, value=0, visible=False)
            
            with gr.Row():
                dd_preprocess_b = gr.Checkbox(label='Inpaint model B detections before model A runs', value=False, visible=False)
                dd_bitwise_op = gr.Radio(label='Bitwise operation', choices=['None', 'A&B', 'A-B'], value="None", visible=False)  
        
        br = gr.HTML("<br>")

        with gr.Group():
            with gr.Row():
                dd_model_b = gr.Dropdown(label="Secondary detection model (B) (optional)", choices=model_list,value = "None", visible =False, type="value")

            with gr.Row():
                dd_conf_b = gr.Slider(label='Detection confidence threshold % (B)', minimum=0, maximum=100, step=1, value=30, visible=False)
                dd_dilation_factor_b = gr.Slider(label='Dilation factor (B)', minimum=0, maximum=255, step=1, value=4, visible=False)
            
            with gr.Row():
                dd_offset_x_b = gr.Slider(label='X offset (B)', minimum=-200, maximum=200, step=1, value=0, visible=False)
                dd_offset_y_b = gr.Slider(label='Y offset (B)', minimum=-200, maximum=200, step=1, value=0, visible=False)
        
        with gr.Group():
            with gr.Row():
                dd_mask_blur = gr.Slider(label='Mask blur ', minimum=0, maximum=64, step=1, value=4, visible=(not is_img2img))
                dd_denoising_strength = gr.Slider(label='Denoising strength (Inpaint)', minimum=0.0, maximum=1.0, step=0.01, value=0.4, visible=(not is_img2img))
            
            with gr.Row():
                dd_inpaint_full_res = gr.Checkbox(label='Inpaint at full resolution ', value=True, visible = (not is_img2img))
                dd_inpaint_full_res_padding = gr.Slider(label='Inpaint at full resolution padding, pixels ', minimum=0, maximum=256, step=4, value=32, visible=(not is_img2img))

        dd_model_a.change(
            lambda modelname: {
                dd_model_b:gr_show( modelname != "None" ),
                dd_conf_a:gr_show( modelname != "None" ),
                dd_dilation_factor_a:gr_show( modelname != "None"),
                dd_offset_x_a:gr_show( modelname != "None" ),
                dd_offset_y_a:gr_show( modelname != "None" )

            },
            inputs= [dd_model_a],
            outputs =[dd_model_b, dd_conf_a, dd_dilation_factor_a, dd_offset_x_a, dd_offset_y_a]
        )

        dd_model_b.change(
            lambda modelname: {
                dd_preprocess_b:gr_show( modelname != "None" ),
                dd_bitwise_op:gr_show( modelname != "None" ),
                dd_conf_b:gr_show( modelname != "None" ),
                dd_dilation_factor_b:gr_show( modelname != "None"),
                dd_offset_x_b:gr_show( modelname != "None" ),
                dd_offset_y_b:gr_show( modelname != "None" )
            },
            inputs= [dd_model_b],
            outputs =[dd_preprocess_b, dd_bitwise_op, dd_conf_b, dd_dilation_factor_b, dd_offset_x_b, dd_offset_y_b]
        )
        
        return [info,
                dd_model_a, 
                dd_conf_a, dd_dilation_factor_a,
                dd_offset_x_a, dd_offset_y_a,
                dd_preprocess_b, dd_bitwise_op, 
                br,
                dd_model_b,
                dd_conf_b, dd_dilation_factor_b,
                dd_offset_x_b, dd_offset_y_b,  
                dd_mask_blur, dd_denoising_strength,
                dd_inpaint_full_res, dd_inpaint_full_res_padding
        ]

    def run(self, p, info,
                     dd_model_a, 
                     dd_conf_a, dd_dilation_factor_a,
                     dd_offset_x_a, dd_offset_y_a,
                     dd_preprocess_b, dd_bitwise_op, 
                     br,
                     dd_model_b,
                     dd_conf_b, dd_dilation_factor_b,
                     dd_offset_x_b, dd_offset_y_b,  
                     dd_mask_blur, dd_denoising_strength,
                     dd_inpaint_full_res, dd_inpaint_full_res_padding):

        processing.fix_seed(p)
        initial_info = None
        seed = p.seed
        p.batch_size = 1
        ddetail_count = p.n_iter
        p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True
        is_txt2img = isinstance(p, StableDiffusionProcessingTxt2Img)
        if (not is_txt2img):
            orig_image = p.init_images[0]
        else:
            p_txt = p
            p = StableDiffusionProcessingImg2Img(
                    init_images = None,
                    resize_mode = 0,
                    denoising_strength = dd_denoising_strength,
                    mask = None,
                    mask_blur= dd_mask_blur,
                    inpainting_fill = 1,
                    inpaint_full_res = dd_inpaint_full_res,
                    inpaint_full_res_padding= dd_inpaint_full_res_padding,
                    inpainting_mask_invert= 0,
                    sd_model=p_txt.sd_model,
                    outpath_samples=p_txt.outpath_samples,
                    outpath_grids=p_txt.outpath_grids,
                    prompt=p_txt.prompt,
                    negative_prompt=p_txt.negative_prompt,
                    styles=p_txt.styles,
                    seed=p_txt.seed,
                    subseed=p_txt.subseed,
                    subseed_strength=p_txt.subseed_strength,
                    seed_resize_from_h=p_txt.seed_resize_from_h,
                    seed_resize_from_w=p_txt.seed_resize_from_w,
                    sampler_name=p_txt.sampler_name,
                    n_iter=p_txt.n_iter,
                    steps=p_txt.steps,
                    cfg_scale=p_txt.cfg_scale,
                    width=p_txt.width,
                    height=p_txt.height,
                    tiling=p_txt.tiling,
                )
            p.do_not_save_grid = True
            p.do_not_save_samples = True
        output_images = []
        state.job_count = ddetail_count
        for n in range(ddetail_count):
            devices.torch_gc()
            start_seed = seed + n
            if ( is_txt2img ):
                print(f"Processing initial image for output generation {n + 1}.")
                p_txt.seed = start_seed
                processed = processing.process_images(p_txt)
                init_image = processed.images[0]   
            else: 
                init_image = orig_image
            
            output_images.append(init_image)
            masks_a = []
            masks_b_pre = []

            # Optional secondary pre-processing run
            if (dd_model_b != "None" and dd_preprocess_b): 
                label_b_pre = "B"
                results_b_pre = inference(init_image, dd_model_b, dd_conf_b/100.0, label_b_pre)
                masks_b_pre = create_segmasks(results_b_pre)
                masks_b_pre = dilate_masks(masks_b_pre, dd_dilation_factor_b, 1)
                masks_b_pre = offset_masks(masks_b_pre,dd_offset_x_b, dd_offset_y_b)
                if (len(masks_b_pre) > 0):
                    results_b_pre = update_result_masks(results_b_pre, masks_b_pre)
                    segmask_preview_b = create_segmask_preview(results_b_pre, init_image)
                    shared.state.current_image = segmask_preview_b
                    if ( opts.dd_save_previews):
                        images.save_image(segmask_preview_b, opts.outdir_ddetailer_previews, "", start_seed, p.prompt, opts.samples_format, p=p)
                    gen_count = len(masks_b_pre)
                    state.job_count += gen_count
                    print(f"Processing {gen_count} model {label_b_pre} detections for output generation {n + 1}.")
                    p.seed = start_seed
                    p.init_images = [init_image]

                    for i in range(gen_count):
                        p.image_mask = masks_b_pre[i]
                        if ( opts.dd_save_masks):
                            images.save_image(masks_b_pre[i], opts.outdir_ddetailer_masks, "", start_seed, p.prompt, opts.samples_format, p=p)
                        processed = processing.process_images(p)
                        p.seed = processed.seed + 1
                        p.init_images = processed.images

                    if (gen_count > 0):
                        output_images[n] = processed.images[0]
                        init_image = processed.images[0]

                else:
                    print(f"No model B detections for output generation {n} with current settings.")

            # Primary run
            if (dd_model_a != "None"):
                label_a = "A"
                if (dd_model_b != "None" and dd_bitwise_op != "None"):
                    label_a = dd_bitwise_op
                results_a = inference(init_image, dd_model_a, dd_conf_a/100.0, label_a)
                masks_a = create_segmasks(results_a)
                masks_a = dilate_masks(masks_a, dd_dilation_factor_a, 1)
                masks_a = offset_masks(masks_a,dd_offset_x_a, dd_offset_y_a)
                if (dd_model_b != "None" and dd_bitwise_op != "None"):
                    label_b = "B"
                    results_b = inference(init_image, dd_model_b, dd_conf_b/100.0, label_b)
                    masks_b = create_segmasks(results_b)
                    masks_b = dilate_masks(masks_b, dd_dilation_factor_b, 1)
                    masks_b = offset_masks(masks_b,dd_offset_x_b, dd_offset_y_b)
                    if (len(masks_b) > 0):
                        combined_mask_b = combine_masks(masks_b)
                        for i in reversed(range(len(masks_a))):
                            if (dd_bitwise_op == "A&B"):
                                masks_a[i] = bitwise_and_masks(masks_a[i], combined_mask_b)
                            elif (dd_bitwise_op == "A-B"):
                                masks_a[i] = subtract_masks(masks_a[i], combined_mask_b)
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
                    segmask_preview_a = create_segmask_preview(results_a, init_image)
                    shared.state.current_image = segmask_preview_a
                    if ( opts.dd_save_previews):
                        images.save_image(segmask_preview_a, opts.outdir_ddetailer_previews, "", start_seed, p.prompt, opts.samples_format, p=p)
                    gen_count = len(masks_a)
                    state.job_count += gen_count
                    print(f"Processing {gen_count} model {label_a} detections for output generation {n + 1}.")
                    p.seed = start_seed
                    p.init_images = [init_image]

                    for i in range(gen_count):
                        p.image_mask = masks_a[i]
                        if ( opts.dd_save_masks):
                            images.save_image(masks_a[i], opts.outdir_ddetailer_masks, "", start_seed, p.prompt, opts.samples_format, p=p)
                        
                        processed = processing.process_images(p)
                        if initial_info is None:
                            initial_info = processed.info
                        p.seed = processed.seed + 1
                        p.init_images = processed.images
                    
                    if (gen_count > 0):
                        output_images[n] = processed.images[0]
                        if ( opts.samples_save ):
                            images.save_image(processed.images[0], p.outpath_samples, "", start_seed, p.prompt, opts.samples_format, info=initial_info, p=p)
  
                else: 
                    print(f"No model {label_a} detections for output generation {n} with current settings.")
            state.job = f"Generation {n + 1} out of {state.job_count}"
        if (initial_info is None):
            initial_info = "No detections found."

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
    if not mmcv_legacy:
        scores = results[3]

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
        if mmcv_legacy:
            score = bboxes[i][4]
        else:
            score = scores[i]
        score = str(score)[:4]
        text = name + ":" + score
        cv2.putText(cv2_image, text, (centroid_x - 30, centroid_y), cv2.FONT_HERSHEY_DUPLEX, 0.4, text_color, 1, cv2.LINE_AA)
    
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

def subtract_masks(mask1, mask2):
    cv2_mask1 = np.array(mask1)
    cv2_mask2 = np.array(mask2)
    cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
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

def offset_masks(masks, offset_x, offset_y):
    if (offset_x == 0 and offset_y == 0):
        return masks
    offset_masks = []
    for i in range(len(masks)):
        cv2_mask = np.array(masks[i])
        offset_mask = cv2_mask.copy()
        offset_mask = np.roll(offset_mask, -offset_y, axis=0)
        offset_mask = np.roll(offset_mask, offset_x, axis=1)
        
        offset_masks.append(Image.fromarray(offset_mask))
    return offset_masks

def combine_masks(masks):
    initial_cv2_mask = np.array(masks[0])
    combined_cv2_mask = initial_cv2_mask
    for i in range(1, len(masks)):
        cv2_mask = np.array(masks[i])
        combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
    
    combined_mask = Image.fromarray(combined_cv2_mask)
    return combined_mask

def on_ui_settings():
    shared.opts.add_option("dd_save_previews", shared.OptionInfo(False, "Save mask previews", section=("ddetailer", "Detection Detailer")))
    shared.opts.add_option("outdir_ddetailer_previews", shared.OptionInfo("extensions/ddetailer/outputs/masks-previews", 'Output directory for mask previews', section=("ddetailer", "Detection Detailer")))
    shared.opts.add_option("dd_save_masks", shared.OptionInfo(False, "Save masks", section=("ddetailer", "Detection Detailer")))
    shared.opts.add_option("outdir_ddetailer_masks", shared.OptionInfo("extensions/ddetailer/outputs/masks", 'Output directory for masks', section=("ddetailer", "Detection Detailer")))

def create_segmasks(results):
    segms = results[2]
    segmasks = []
    for i in range(len(segms)):
        cv2_mask = segms[i].astype(np.uint8) * 255
        mask = Image.fromarray(cv2_mask)
        segmasks.append(mask)

    return segmasks

import mmcv

try:
    from mmdet.core import get_classes
    from mmdet.apis import (inference_detector,
                        init_detector)
    mmcv_legacy = True
except ImportError:
    from mmdet.evaluation import get_classes
    from mmdet.apis import inference_detector, init_detector
    mmcv_legacy = False

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
    if mmcv_legacy:
        model = init_detector(model_config, model_checkpoint, device=model_device)
        mmdet_results = inference_detector(model, np.array(image))
        bbox_results, segm_results = mmdet_results
    else:
        model = init_detector(model_config, model_checkpoint, palette="random", device=model_device)
        mmdet_results = inference_detector(model, np.array(image)).pred_instances
        bboxes = mmdet_results.bboxes.numpy()

    dataset = modeldataset(modelname)
    classes = get_classes(dataset)
    if mmcv_legacy:
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_results)
        ]
        n, m = bbox_results[0].shape
    else:
        n, m = bboxes.shape
    if (n == 0):
        if mmcv_legacy:
            return [[],[],[]]
        else:
            return [[],[],[],[]]

    if mmcv_legacy:
        labels = np.concatenate(labels)
        bboxes = np.vstack(bbox_results)
        segms = mmcv.concat_list(segm_results)

        filter_inds = np.where(bboxes[:,-1] > conf_thres)[0]
        results = [[],[],[]]
    else:
        labels = mmdet_results.labels
        segms = mmdet_results.masks.numpy()
        scores = mmdet_results.scores.numpy()

        filter_inds = np.where(mmdet_results.scores > conf_thres)[0]
        results = [[],[],[],[]]

    for i in filter_inds:
        results[0].append(label + "-" + classes[labels[i]])
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        if not mmcv_legacy:
            results[3].append(scores[i])

    return results

def inference_mmdet_bbox(image, modelname, conf_thres, label):
    model_checkpoint = modelpath(modelname)
    model_config = os.path.splitext(model_checkpoint)[0] + ".py"
    model_device = get_device()

    if mmcv_legacy:
        model = init_detector(model_config, model_checkpoint, device=model_device)
        results = inference_detector(model, np.array(image))
    else:
        model = init_detector(model_config, model_checkpoint, device=model_device, palette="random")
        output = inference_detector(model, np.array(image)).pred_instances
    cv2_image = np.array(image)
    cv2_image = cv2_image[:, :, ::-1].copy()
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    bboxes = []
    if mmcv_legacy:
        for (x0, y0, x1, y1, conf) in results[0]:
            bboxes.append([x0, y0, x1, y1])
    else:
        bboxes = output.bboxes

    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros((cv2_gray.shape), np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    if mmcv_legacy:
        n,m = results[0].shape
    else:
        n,m = output.bboxes.shape
    if (n == 0):
        if mmcv_legacy:
            return [[],[],[]]
        else:
            return [[],[],[],[]]
    if mmcv_legacy:
        bboxes = np.vstack(results[0])
        filter_inds = np.where(bboxes[:,-1] > conf_thres)[0]
        results = [[],[],[]]
    else:
        bboxes = output.bboxes.numpy()
        scores = output.scores.numpy()
        filter_inds = np.where(scores > conf_thres)[0]
        results = [[],[],[],[]]

    for i in filter_inds:
        results[0].append(label)
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        if not mmcv_legacy:
            results[3].append(scores[i])

    return results

script_callbacks.on_ui_settings(on_ui_settings)
