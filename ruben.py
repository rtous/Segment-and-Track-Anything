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
import pickle

def main(scene, keyword_lists):

	print("Ruben's version of SAM-Track") 
	#file_path = "data/scenes/tiktok2/imagesFull"
	#output_path = "data/scenes/tiktok2/samtrack"
	
	#file_path = "data/scenes/assault2_1/imagesFull"
	#output_path = "data/scenes/assault2_1/samtrack"

	file_path = "data/scenes/"+scene+"/imagesFull"
	output_path = "data/scenes/"+scene+"/samtrack"
	########################
	#grounding_caption = "skin, tshirt, hair, ball, legs"
	#grounding_caption = "trousers, skin, tshirt, hair, ball" FATAL
	#grounding_caption = "arms, face, tshirt, trousers, hair, ball"
	#grounding_caption = "hair, arms, legs, tshirt, ball" FATAL
	#grounding_caption = "trousers, tshirt, ball, skin, hair" FATAL
	#grounding_caption = "face, hair, arms, legs, tshirt, ball" BARBA!
	#grounding_caption = "skin, hair, legs, tshirt, ball" #OK NO HAIR
	#grounding_caption = "skin, face, hair, legs, tshirt, ball" NO tshirt
	#grounding_caption = "hair, skin, legs, tshirt, ball" FAIL
	#grounding_caption = "skin, legs, arms, tshirt, ball, hair" OK, NO HAIR but braços diferents
	#grounding_caption = "face, legs, arms, tshirt, ball, hair" FAIL
	#grounding_caption = "legs, skin, tshirt, ball, hair" FAIL
	#grounding_caption = "skin, legs, arms, tshirt, ball, hair" #OK NO HAIR
	#grounding_caption = "face, hair, arms, legs, tshirt, ball"
	grounding_caption = "skin, hair"  
	#NOT USED NOW
	#########################
	

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


	#1) Initialize a SegTracker with the first frame

	origin_frame = first_frame
	aot_model = "r50_deaotl"# "deaotb", "deaotl", "r50_deaotl" (default "r50_deaotl")                                 
	long_term_mem = 9999 #1-9999 (default 9999)
	max_len_long_term = 9999 #1-9999 (default 9999)
	sam_gap = 100 #1-9999 (default 100)
	max_obj_num = 50 #50-300 (default 255)
	points_per_side = 16 #1-100 (default 16)

	#Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)
	'''
	print("Segmenting first frame...")
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


	print("Done segmenting first frame, result written to file")

	cv2.imwrite(os.path.join(output_path, os.path.basename(imgs_paths[0])), masked_frame)
	'''

	#Lists of keywords
	#First list will be added first, so put details the last
	#keyword_lists = ["hair", "skin, legs, arms, tshirt, ball"] #ok ruben pilota
	#keyword_lists = ["hair, shoes", "skin, legs, arms, shirt, skirt"] #ok green woman 1
	#keyword_lists = ["hair, shoes", "skin, legs, arms, clothes"] #ok green woman 2
	#keyword_lists = ["hair", "skin, jacket, shirt"] #ok green woman 3
	#keyword_lists = ["lights, windows, tires", "car"] #ok car
	#keyword_lists = ["hair, shoes", "skin, trousers, tshirt"] #ok arizona1
	#keyword_lists = ["hair", "skin, tshirt"] #testing arizona3
	#keyword_lists = ["hair, shoes", "skin, trousers, tshirt"] #testing arizona4_part1
	#keyword_lists = ["hair, feet", "skin, trousers, tshirt"] #testing arizona5
	#keyword_lists = ["hair, shoes", "skin, trousers, tshirt"] #testing man_walk_1_part1

	print("Working with keyword_lists:", keyword_lists)
	video_name = "example"
	for i, keyword_list in enumerate(keyword_lists):

		Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)


		grounding_caption = keyword_list

		#2) Detect objects by text (grounding_caption) over the first frame")
		print("Detecting objects by text...")
		text_threshold = 0.5#0.25
		box_threshold = 0.2#0.25
		
		predicted_mask, annotated_frame= Seg_Tracker.detect_and_seg(origin_frame, grounding_caption, box_threshold, text_threshold)
		Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)



		#Despres es crida a tracking_objects_in_video del fitxer seg_track_anything
		#tracking_objects_in_video(Seg_Tracker, input_video, input_img_seq, fps, frame_num)

		#o millor directament a img_seq_type_input_tracking
		fps = 8 #Web UI
		frame_num=0

		#file_name = input_img_seq.name.split('/')[-1].split('.')[0]
		imgs_path = sorted([os.path.join(file_path, img_name) for img_name in os.listdir(file_path)])
		
		io_args = {
		        'tracking_result_dir': output_path,
		        'output_mask_dir': f'{output_path}/{video_name}_masks{i}',
		        'output_masked_frame_dir': f'{output_path}/{video_name}_masked_frames{i}',
		        'output_video': f'{output_path}/{video_name}_seg{i}.mp4', # keep same format as input video
		        'output_gif': f'{output_path}/{video_name}_seg{i}.gif',
		    }
		print("Segmenting...")
		img_seq_type_input_tracking(Seg_Tracker, io_args, video_name, imgs_path, fps, frame_num)
		print("Done.")

	all_dir = f'{output_path}/{video_name}_all'
	if not os.path.exists(all_dir):
		os.mkdir(all_dir)
	for i, keyword_list in enumerate(keyword_lists):
		masks_dir = f'{output_path}/{video_name}_masks{i}'
		for filename in os.listdir(masks_dir):
			path_mask = os.path.join(masks_dir, filename)
			path_all = os.path.join(all_dir, filename)
			if i == 0:
				all_masks = cv2.imread(path_mask)
			else:
				all_masks = cv2.imread(path_all)
				mask = cv2.imread(path_mask)
				all_masks = cv2.addWeighted(all_masks,1.0,mask,1.0,0)
			cv2.imwrite(path_all, all_masks)


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
    print("Ruben's SAM-Track launcher")
    print("scene:", sys.argv[1])
    print("keyword lists:", pickle.loads(sys.argv[2]))
    main(sys.argv[1], pickle.loads(sys.argv[2]))