import time
import numpy as np

from . import geo

def has_gps_info(exif):
    return (exif
            and 'gps' in exif
            and 'latitude' in exif['gps']
            and 'longitude' in exif['gps'])


def distance_from_exif(exif1, exif2):
    """Compute distance between images based on exif metadata.

    >>> exif1 = {'gps': {'latitude': 50.0663888889, 'longitude': 5.714722222}}
    >>> exif2 = {'gps': {'latitude': 58.6438888889, 'longitude': 3.070000000}}
    >>> d = distance_from_exif(exif1, exif2)
    >>> abs(d - 968998) < 1
    True
    """
    if has_gps_info(exif1) and has_gps_info(exif2):
        gps1 = exif1['gps']
        gps2 = exif2['gps']
        latlon1 = gps1['latitude'], gps1['longitude']
        latlon2 = gps2['latitude'], gps2['longitude']
        return geo.gps_distance(latlon1, latlon2)
    else:
        return 0


def timediff_from_exif(exif1, exif2):
    return np.fabs(exif1['capture_time'] - exif2['capture_time'])


def match_candidates_from_metadata(images, exifs, data):
    '''
    Compute candidate matching pairs based on GPS and capture time
    '''
    max_distance = data.config['matching_gps_distance']
    max_neighbors = data.config['matching_gps_neighbors']
    max_time_neighbors = data.config['matching_time_neighbors']
    max_order_neighbors = data.config['matching_order_neighbors']

    if not all(map(has_gps_info, exifs.values())) and max_neighbors != 0:
        logger.warn("Not all images have GPS info. "
                    "Disabling matching_gps_neighbors.")
        max_neighbors = 0

    pairs = set()
    images.sort()
    for index1, im1 in enumerate(images):
        distances = []
        timediffs = []
        indexdiffs = []
        for index2, im2 in enumerate(images):
            if im1 != im2:
                dx = distance_from_exif(exifs[im1], exifs[im2])
                dt = timediff_from_exif(exifs[im1], exifs[im2])
                di = abs(index1 - index2)
                if dx <= max_distance:
                    distances.append((dx, im2))
                    timediffs.append((dt, im2))
                    indexdiffs.append((di, im2))
        distances.sort()
        timediffs.sort()
        indexdiffs.sort()

        if max_neighbors or max_time_neighbors or max_order_neighbors:
            distances = distances[:max_neighbors]
            timediffs = timediffs[:max_time_neighbors]
            indexdiffs = indexdiffs[:max_order_neighbors]

        for d, im2 in distances + timediffs + indexdiffs:
            if im1 < im2:
                pairs.add((im1, im2))
            else:
                pairs.add((im2, im1))

    res = {im: [] for im in images}
    for im1, im2 in pairs:
        res[im1].append(im2)
        res[im2].append(im1)
        
    return res


def match_arguments(pairs, ctx):
    for i, (im, candidates) in enumerate(pairs.items()):
        yield im, candidates, i, len(pairs), ctx

