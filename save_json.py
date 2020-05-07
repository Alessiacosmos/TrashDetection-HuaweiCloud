# -*- encoding: utf-8 -*-
"""
@File    : save_json.py
@Time    : 2020/5/4 18:08
@Author  : Alessia K
@Email   : ------
"""
import json
import numpy as np
import os
import csv


def label_list(path):
	with open(path, 'r', encoding='utf-8') as f:
		class_csv = csv.reader(f, delimiter=',')
		result = {}

		for line, row in enumerate(class_csv):
			line += 1

			try:
				class_name, class_id = row
			except ValueError:
				# raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
				raise (ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)

			if class_name in result:
				raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
			result[int(class_id)] = class_name
		return result

def create_class_dict(path):
	with open(os.path.join(path, 'classify_rule.json'), 'r', encoding='utf-8') as f:
		classify_rule = json.load(f)
		class_name = {}
		for keys, vals in classify_rule.items():
			for i in range(len(vals)):
				class_name[str(vals[i])]=str(keys)

	return class_name


def get_classes_name(class_name, class_name_i):
	return class_name[class_name_i] + '/' + class_name_i


def save_result_as_json(img_name, classes, scores, bboxes, time):
	"""
	{
	    "detection_classes":    []
	    "detection_scores":     []  (.4f)
	    "detection_bboxes":     []  (xmin、ymin、xmax、ymax) (.1f)
	    "latency_time":         ""  (str(.1f))
	}
	"""

	scores = np.around(scores.astype(np.float), decimals=4).tolist()
	bboxes = np.around(bboxes.astype(np.float), decimals=1).tolist()
	print(bboxes)


	save_file = {}
	save_file["detection_classes"] 	= classes
	save_file["detection_scores"] 	= scores
	save_file["detection_boxes"] 	= bboxes
	save_file["latency_time"] 		= "{:.1f} ms".format(time)

	return save_file

	# save_name = os.path.join('data', os.path.basename(img_name).replace('.jpg', '.json'))
	# with open(save_name, 'w+', encoding='utf-8')as f:
	# 	json.dump(save_file, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
	class_name = create_class_dict(r'D:\Work\ohter\SodicData\train_val')
	print(class_name)
	labels = label_list('data/class_name.csv')
	print(labels)