import argparse, os, glob, json

from common import get_images_list, get_exifs
from opensfm.match_features import match_candidates_from_metadata

parser = argparse.ArgumentParser(description='Extracts EXIF data from set of images')
parser.add_argument('-i', '--images', nargs='+', metavar="<path or wildcard pattern>", required=True, help='Input images path (directory or wildcard pattern)')
parser.add_argument('--format', metavar="<string>", default="csv", help='Output format of result. One of: csv Default: %(default)s')
args = parser.parse_args()

images_list = get_images_list(args.images)
exifs = get_exifs(images_list)

print("""#F=N Y X Z
#
#image	latitude	longitude	altitude""")

for im, exif in exifs.items():
	print("{}	{}	{}	{}".format(os.path.basename(im), exif['gps']['latitude'], exif['gps']['longitude'], exif['gps']['altitude']))

