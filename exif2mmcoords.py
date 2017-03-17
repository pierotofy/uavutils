import argparse, os, glob, json

from common import get_images_list, get_exifs
from opensfm.match_features import match_candidates_from_metadata

parser = argparse.ArgumentParser(description='Generate a micmac coordinate file from EXIF image(s)')
parser.add_argument('-i', '--images', nargs='+', metavar="<path or wildcard pattern>", required=True, help='Input images path (directory or wildcard pattern)')
args = parser.parse_args()

images_list = get_images_list(args.images)
exifs = get_exifs(images_list)
first_exif = next (iter (exifs.values()))

print("""<SystemeCoord>
     <BSC>
        <TypeCoord>eTC_RTL</TypeCoord>
        <AuxR>{}</AuxR>
        <AuxR>{}</AuxR>
        <AuxR>{}</AuxR>
     </BSC>
     <BSC>
        <TypeCoord>eTC_WGS84</TypeCoord>
        <AuxRUnite>eUniteAngleDegre</AuxRUnite>
     </BSC>
</SystemeCoord>""".format(first_exif['gps']['latitude'], first_exif['gps']['longitude'], first_exif['gps']['altitude']))