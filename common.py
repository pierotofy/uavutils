import os, glob

from opensfm import exif

def get_images_list(images_patterns):
	images_list = []
	for image in images_patterns:
		if os.path.isdir(image):
			for ext in ["JPG", "JPEG", "jpg", "jpeg"]:
				images_list += glob.glob(os.path.join(image, "*.{}".format(ext)))
		elif os.path.isfile(image):
			images_list += [image]

	return images_list

def get_exifs(images_list):
	exifs = {}
	for image_path in images_list:
		exifs[image_path] = exif.extract_exif_from_file(open(image_path, 'rb'))
	return exifs