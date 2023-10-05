from PIL.ImageOps import colorize, scale
#import gradio as gr
import importlib
import sys
import os
import pdb
from matplotlib.pyplot import step

from model_args import segtracker_args,sam_args,aot_args
from SegTracker import SegTracker
from tool.transfer_tools import draw_outline, draw_points
# sys.path.append('.')
# sys.path.append('..')

import cv2
from PIL import Image
from skimage.morphology.binary import binary_dilation
import argparse
import torch
import time, math
from seg_track_anything import aot_model2ckpt, tracking_objects_in_video, draw_mask, img_seq_type_input_tracking
import gc
import numpy as np
import json
from tool.transfer_tools import mask2bbox

def main():

	print("Ruben's version of SAM-Track") 
	file_path = "data/scenes/tiktok2/imagesFull"
	output_path = "data/scenes/tiktok2/samtrack"
	imgs_paths = sorted([os.path.join(file_path, img_name) for img_name in os.listdir(file_path) if img_name.endswith(".png") or img_name.endswith(".jpg")])
	
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	#blank_image = np.zeros((256,256,3), np.uint8)
	#cv2.imwrite(os.path.join(output_path, "test.png"), blank_image)


	print("Reading first frame from ", imgs_paths[0])
	first_frame_file_path = imgs_paths[0]
	first_frame = cv2.imread(first_frame_file_path)
	first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

	#cv2.imwrite(os.path.join(output_path, first_frame_file_name), blank_image)


	origin_frame = first_frame
	aot_model = "r50_deaotl"# "deaotb", "deaotl", "r50_deaotl" (default "r50_deaotl")                                 
	long_term_mem = 9999 #1-9999 (default 9999)
	max_len_long_term = 9999 #1-9999 (default 9999)
	sam_gap = 100 #1-9999 (default 100)
	max_obj_num = 50 #50-300 (default 255)
	points_per_side = 16 #1-100 (default 16)

	Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)

	print("Everything")

	frame_idx = 0

	with torch.cuda.amp.autocast():
		pred_mask = Seg_Tracker.seg(origin_frame)
		torch.cuda.empty_cache()
		gc.collect()
		Seg_Tracker.add_reference(origin_frame, pred_mask, frame_idx)
		Seg_Tracker.first_frame_mask = pred_mask

	just_mask = np.zeros_like(origin_frame)
	#masked_frame = draw_mask(origin_frame.copy(), pred_mask)
	masked_frame = draw_mask(just_mask, pred_mask)


	print("Done, writing to file...")

	cv2.imwrite(os.path.join(output_path, os.path.basename(imgs_paths[0])), masked_frame)


	#A a UI primer es crida a Detect

	text_threshold = 0.25
	box_threshold = 0.25
	grounding_caption = "skin"
	predicted_mask, annotated_frame= Seg_Tracker.detect_and_seg(origin_frame, grounding_caption, box_threshold, text_threshold)

	Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)



	#Despres es crida a tracking_objects_in_video del fitxer seg_track_anything
	#tracking_objects_in_video(Seg_Tracker, input_video, input_img_seq, fps, frame_num)

	#o millor directament a img_seq_type_input_tracking
	fps = 8 #Web UI
	frame_num=0

	#file_name = input_img_seq.name.split('/')[-1].split('.')[0]
	file_path = f'data/scenes/tiktok2/imagesFull'
	imgs_path = sorted([os.path.join(file_path, img_name) for img_name in os.listdir(file_path)])
	video_name = "example"
	io_args = {
	        'tracking_result_dir': output_path,
	        'output_mask_dir': f'{output_path}/{video_name}_masks',
	        'output_masked_frame_dir': f'{output_path}/{video_name}_masked_frames',
	        'output_video': f'{output_path}/{video_name}_seg.mp4', # keep same format as input video
	        'output_gif': f'{output_path}/{video_name}_seg.gif',
	    }
	img_seq_type_input_tracking(SegTracker, io_args, video_name, imgs_path, fps, frame_num)

def init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame):
    
    if origin_frame is None:
        return None, origin_frame, [[], []], ""

    # reset aot args
    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]
    aot_args["long_term_mem_gap"] = long_term_mem
    aot_args["max_len_long_term"] = max_len_long_term
    # reset sam args
    segtracker_args["sam_gap"] = sam_gap
    segtracker_args["max_obj_num"] = max_obj_num
    sam_args["generator_args"]["points_per_side"] = points_per_side
    
    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()

    return Seg_Tracker, origin_frame, [[], []], ""

def SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask):
    with torch.cuda.amp.autocast():
        # Reset the first frame's mask
        frame_idx = 0
        Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)
        Seg_Tracker.first_frame_mask = predicted_mask

    return Seg_Tracker



if __name__ == "__main__":
    main()