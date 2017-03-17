import argparse, os, glob, json

from common import get_images_list, get_exifs
from opensfm.match_features import match_candidates_from_metadata

parser = argparse.ArgumentParser(description='Converts a set of images with embedded EXIF GPS data into a list of matching neighbors')
parser.add_argument('-i', '--images', nargs='+', metavar="<path or wildcard pattern>", required=True, help='Input images path (directory or wildcard pattern)')
parser.add_argument('--matching_gps_neighbors', metavar="<num>", type=int, default=8, help='Number of images to match selected by GPS distance. Set to 0 to use no limit. Default: %(default)s')
parser.add_argument('--matching_gps_distance', metavar="<num>", type=int, default=150, help='Maximum gps distance between two images for matching. Default: %(default)s')
parser.add_argument('--matching_time_neighbors', metavar="<num>", type=int, default=0, help='Number of images to match selected by time taken. Set to 0 to use no limit. Default: %(default)s')
parser.add_argument('--matching_order_neighbors', metavar="<num>", type=int, default=0, help='Number of images to match selected by image name. Set to 0 to use no limit. Default: %(default)s')
parser.add_argument('--format', metavar="<string>", default="plain", help='Output format of result. One of: plain|micmac Default: %(default)s')

args = parser.parse_args()

images_list = get_images_list(args.images)

class Object(object):
    pass
data = Object()
data.config = {
	'matching_gps_distance': args.matching_gps_distance,
	'matching_gps_neighbors': args.matching_gps_neighbors,
	'matching_time_neighbors': args.matching_time_neighbors,
	'matching_order_neighbors': args.matching_order_neighbors
}
exifs = get_exifs(images_list)

matches = match_candidates_from_metadata(images_list, exifs, data)
if args.format == 'micmac':
	print("<?xml version=\"1.0\" ?>\n"
"<SauvegardeNamedRel>")

	for image in matches:
		for match in matches[image]:
			print("	<Cple>{} {}</Cple>".format(os.path.basename(image), os.path.basename(match)))

	print("</SauvegardeNamedRel>\n")
else:
	for image in matches:
		for match in matches[image]:
			print("{} {}".format(os.path.basename(image), os.path.basename(match)))
