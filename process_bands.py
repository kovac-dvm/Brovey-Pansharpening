import numpy as np

from PIL.Image import open as PilOpen
from rasterio import open as RastOpen
from rasterio.warp import calculate_default_transform

import cv2

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

PilOpen.MAX_IMAGE_PIXELS = 200000000000

# In[] Description

""""
The Brovey transformation is used as a pan-sharpening method.

Function "get_cube" performs pan-sharpening of low-resolution data from multi-spectral camera based on the
high-resolution RGB image. If the pixel size of the data from a multi-spectral camera is not uniform compared to that
of a RGB image, up-sampling is performed. The function returns a three-dimensional ndarray of type float32 and
generates georeferenced .TIF file and optionally a .PDF file for analyzing specific bands and pan-sharpening results.

Other functions can be used separately.
"""

# In[] Variables (example if running as the main script)

if __name__ == '__main__':
    ID = "755"
    file_path_rgb = "data/image-tiles/rgb_" + ID + ".png"
    file_path_multi = "data/image-tiles/multispectral_" + ID + ".tif"
    channel_order = ['blue', 'green', 'red', 'red-edge', 'near-IR']
    georeferences = True
    interpolation_method = 'bicubic'
    get_tif_file = True
    get_pic = True


# In[] Main function
def get_cube(rgb_data: any,
             multi_band_data: any,
             band_order: list = None,
             interpolation: str = 'bilinear',
             get_tif: bool = True,
             georeference_image: bool = True,
             file_name: str = 'my_sharp_image',
             get_picture: bool = False):
    """
    Merges low-resolution channels from multi-spectral camera with data from high-resolution RGB bands,
    and perform pan-sharpening.

    :param rgb_data:        ndarray (height, width, number of bands) in a float32 format
                            or string path to RGB image file in .png or .jpg format (red, green, blue)
    :param multi_band_data:      ndarray (height, width, number of bands) in a float32 format
                            or string path to multi-spectral image file in .tif format (should contain RGB channels)
    :param band_order:      sorted list of channel names in a multi-spectral image, RGB channels should be named as
                            either the full color name or just the initial letter, both uppercase/lowercase letters work
                            - e.g. ['bLuE', 'G', 'r', 'red-edge', 'near-IR'],
    :param interpolation:   interpolation method: bilinear (default), nearest, area, bicubic, lanczos
    :param get_tif:         save cube as .TIF file with the original metadata
                            referenced by it, if not, the image will be referenced by the input MS TIF
    :param georeference_image:  georeference image with the original metadata
    :param file_name:            output file name or existing directory with the name of new a file
    :param get_picture:     save image with a comparison of specific bands as .pdf file


    :return:                ndarray (height, width, number of bands) in a float32 format
                            - high-resolution RGB channels with pan-sharped channels from multi-spectral camera in order
                            given by the parameter band_order
                            - file_name
    :rtype:                 ndarray, str
    """

    # load data
    if isinstance(rgb_data, str):
        array_rgb = load_rgb(rgb_data)
    else:
        array_rgb = rgb_data

    if isinstance(multi_band_data, str):
        array_multi, original_metadata = load_multi(multi_band_data)
    else:
        array_multi = multi_band_data
        original_metadata = None

    # create a one-dimensional panchromatic array from a high-resolution RGB array
    array_rgb_panchromatic = get_panchromatic(array_rgb)

    # up-sample multi-spectral channels to fit the size of the RGB image
    if not (array_rgb.shape[0], array_rgb.shape[1]) == (array_multi.shape[0], array_multi.shape[1]):
        array_multi_up = up_sample(array_multi, (array_rgb.shape[0], array_rgb.shape[1]), interpolation)
    else:
        array_multi_up = array_multi

    # perform pan-sharpening of non-RGB multi-spectral channels based on panchromatic image of higher-resolution RGB
    array_multi_sharp = pan_sharpening(array_rgb_panchromatic, array_multi_up)

    # create and save a .tif file
    if get_tif:
        save_tif(array_multi_sharp,
                 file_name=f"{file_name}.tif",
                 georeferecning=georeference_image,
                 original_ms_data_path=multi_band_data)

    # create and save a .pdf file to analyze the data and results of pan-sharpening
    if get_picture:
        save_picture(file_name, band_order, array_rgb, array_rgb_panchromatic, array_multi_up, array_multi_sharp)

    return array_multi_sharp, f"{file_name}.tif"


# In[] Other (necessary) functions

def load_rgb(pth_rgb: str):
    """
    Load the high-resolution RGB image file in .png or .jpg format (red, green, blue).
    :param pth_rgb: path to the file
    :return:        ndarray (height, width, number of bands) in a float32 format
    """
    tile_rgb = PilOpen(pth_rgb)
    rgb = np.asarray(tile_rgb, dtype=np.float32) / 255
    return rgb


def load_multi(pth_multi: str):
    """
    Load the low-resolution multi-spectral image file in .tif format.
    :param pth_multi:   path to the file
    :return:            ndarray (height, width, number of bands) in a float32 format
    """
    tile_multi = RastOpen(pth_multi)
    meta = tile_multi.meta
    multi = tile_multi.read()
    multi_reshaped = np.swapaxes(multi, 0, 2)
    multi_reshaped2 = np.swapaxes(multi_reshaped, 0, 1)
    return multi_reshaped2, meta


def get_idx(bands: list):
    """
    Get indexes of RGB channels and the rest of bands from multi-spectral image.
    :param bands:   sorted list of channel names in a multi-spectral image, RGB channels should be named as either
                    the full color name or just the initial letter, both uppercase/lowercase letters work
                    - e.g. ['bLuE', 'G', 'r', 'red-edge', 'near-IR'],
    :return:        tuple of lists - the first of RGB positions in the image and the second of the rest of the bands
    """
    idx_b = [i for i in range(len(bands)) if bands[i].upper() == 'B' or bands[i].upper() == 'BLUE']
    if len(idx_b) > 1:
        raise Exception("There are more bands in the list defined as blue.")
    elif len(idx_b) == 0:
        raise Exception("There should be at least one band in the list defined as blue.")
    else:
        idx_b = idx_b[0]
    idx_g = [i for i in range(len(bands)) if bands[i].upper() == 'G' or bands[i].upper() == 'GREEN']
    if len(idx_g) > 1:
        raise Exception("There are more bands in the list defined as green.")
    elif len(idx_g) == 0:
        raise Exception("There should be at least one band in the list defined as green.")
    else:
        idx_g = idx_g[0]
    idx_r = [i for i in range(len(bands)) if bands[i].upper() == 'R' or bands[i].upper() == 'RED']
    if len(idx_r) > 1:
        raise Exception("There are more bands in the list defined as red.")
    elif len(idx_r) == 0:
        raise Exception("There should be at least one band in the list defined as red.")
    else:
        idx_r = idx_r[0]
    index_rgb = list([idx_r, idx_g, idx_b])
    index_rest = [k for k in range(len(bands)) if k not in list(index_rgb)]
    return index_rgb, index_rest


def up_sample(image: object, dimension: tuple, method='bilinear'):
    """
    Up-sample image to a desired shape.
    :param image:       openCV image or ndarray (height, width, number of bands) in a float32 format
    :param dimension:   height and width of the resulting image
    :param method:      interpolation method: bilinear (default), nearest, area, bicubic, lanczos
    :return:            up-sampled openCV image  or ndarray (height, width, number of bands) in a float32 format
    """

    methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }

    try:
        inter = methods[method]
    except KeyError:
        raise Exception('Unknown interpolation method.')

    resampled_img = cv2.resize(image, (dimension[0], dimension[1]), interpolation=inter)
    return resampled_img


def switch_rgb(rgb: object):
    """
    Get BGR from RGB (or vice-versa).
    :param rgb: ndarray (height, width, number of bands) in a float32 format
    :return:    ndarray (height, width, number of bands) in a float32 format
    """
    bgr = rgb[:, :, ::-1]
    return bgr


def cut_multi(multi: object, idxs: tuple):
    """
    Separate channels from multi-spectral image.
    :param multi:   ndarray (height, width, number of bands) in a float32 format
    :param idxs:    tuple of lists - the first of RGB positions in the image and the second of the rest of the bands
    :return:        2x ndarray (height, width, number of bands) in a float32 format:
                    (red, green, blue), (rest of the channels)
    """
    rgb = multi[:, :, idxs[0]]
    rest = multi[:, :, idxs[1]]
    return rgb, rest


def get_panchromatic(rgb: object):
    """
    Get a pan-chromatic image from the RGB image.
    :param rgb: ndarray (height, width, number of bands) in a float32 format
    :return:    ndarray (height, width, 1) in a float32 format
    """
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return grey


def pan_sharpening(grey: object, multi: object):
    """
    Perform pan-sharpening based on the Brovey transformation method.
    :param grey:    ndarray of pan-chromatic or greyscale image (height, width, 1) in a float32 format
    :param multi:   ndarray (height, width, number of bands) in a float32 format - the number of channels is arbitrary
    :return:        ndarray (height, width, number of bands) in a float32 format
    """
    weights = (grey / np.sum(multi, axis=2))
    pan_sharpened = np.zeros_like(multi)
    for band in range(multi.shape[2]):
        pan_sharpened[:, :, band] = multi[:, :, band] * weights
    return pan_sharpened


def merge_bands(band_rgb: object, band_rest: object, idxs: tuple):
    """
    Merge RGB array bands and rest channels of multi-spectral image according to their positions
    given by parameter idxs.
    :param band_rgb:    ndarray (height, width, number of bands) in a float32 format
    :param band_rest:   ndarray (height, width, number of bands) in a float32 format
    :param idxs:        tuple of lists - the first of RGB positions in the image and the second of the rest of the bands
    :return:            merged ndarray (height, width, number of bands) in a float32 format
    """
    merged = np.zeros((band_rgb.shape[0], band_rgb.shape[1], max(max(idxs)) + 1), dtype=band_rest.dtype)
    merged[:, :, idxs[0]] = band_rgb
    merged[:, :, idxs[1]] = band_rest
    return merged


def save_picture(name, bands, rgb, rgb_panchromatic, multi_up, multi_sharp):
    """
    Perform pan-sharpening of RGB channels from multi-spectral camera and save image as .pdf file.
    :param name:                output file name
    :param bands:               sorted list of channel names in a multi-spectral image, RGB channels should be named as
                                the full color name or just the initial letter, both uppercase/lowercase letters work
                                - e.g. ['bLuE', 'G', 'r', 'red-edge', 'near-IR'],
    :param rgb:                 ndarray (height, width, 3) in a float32 format (high-resolution RGB)
    :param rgb_panchromatic:    ndarray (height, width, 1) in a float32 format (high-resolution grey)
    :param multi_up:            ndarray (height, width, 2) in a float32 format (up-sampled)
    :param multi_sharp:         ndarray (height, width, 2) in a float32 format (pan-sharped)
    :return:                    .pdf file
    """

    if bands is None:
        bands = [f"Channel {i + 1}" for i in range(multi_up.shape[2])]

    n_pics_in_figure = 4
    size = (22, 9)
    n_pics = multi_up.shape[2] + 1
    n_figures = 0
    for n in range(n_pics):
        if n > 0 and n % n_pics_in_figure == 0:
            n_figures += 1
            plt.figure(n_figures + 1, figsize=size)
        position_low = n + 1 - (n_pics_in_figure * n_figures)
        position_sharp = position_low + n_pics_in_figure
        if n == 0:
            plt.figure(1, figsize=size)
            plt.subplot(2, n_pics_in_figure, position_low)
            plt.imshow(rgb)
            plt.title("RGB (high-res)")
            plt.subplot(2, n_pics_in_figure, position_sharp)
            plt.imshow(rgb_panchromatic, cmap='gray')
            plt.title("Panchromatic from RGB (high-res)")
        else:
            plt.subplot(2, n_pics_in_figure, position_low)
            plt.imshow(multi_up[:, :, n-1])
            plt.title(bands[n-1] + " from multi-spectral camera")
            plt.subplot(2, n_pics_in_figure, position_sharp)
            plt.imshow(multi_sharp[:, :, n-1])
            plt.title("Pan-sharped " + bands[n-1] + " from multi-spectral camera")
    with PdfPages(name + '.pdf') as pdf:
        for page in range(n_figures+1):
            pdf.savefig(plt.figure(page+1))


def save_tif(new_array_data,
             file_name: str,
             georeferecning: bool = True,
             original_ms_data_path: str = None,
             ):
    """
    Save array as a .TIF file.
    :param new_array_data:   ndarray (height, width, number of channels)
    :param georeferecning:  if True, georeferencing is performed
    :param original_ms_data_path: path to original ms data, in order to read file information
    :param file_name:    output file name or existing directory with the name of new a file
    :return:        .TIF file
    """

    # If geo-referencing is not required, save the file without geo-referencing
    if not georeferecning:

        # Prepare new metadata
        metadata = {
            'driver': 'GTiff',
            'height': new_array_data.shape[0],
            'width': new_array_data.shape[1],
            'count': new_array_data.shape[2],
            'dtype': new_array_data.dtype
        }

    # Recalculate Affine transformation and dimensions of the destination raster
    else:
        # Read metadata of original ms data
        with RastOpen(original_ms_data_path) as src:
            original_metadata = src.meta.copy()

        # Calculate the transform and dimensions of the destination raster
        source_crs = src.crs
        # have to be the same in this case
        destination_crs = source_crs

        transform, width, height = calculate_default_transform(source_crs,
                                                               destination_crs,
                                                               src.width,
                                                               src.height,
                                                               *src.bounds,
                                                               dst_width=new_array_data.shape[1],
                                                               dst_height=new_array_data.shape[0])
        # Store new metadata
        metadata = original_metadata.copy()
        metadata.update({
            'transform': transform,
            'width': width,
            'height': height
        })

    # Save the file
    with RastOpen(file_name, 'w', **metadata) as dst:
        for band in range(1, dst.count):
            dst.write(new_array_data[:, :, band - 1], band)


# In[] Run

if __name__ == '__main__':
    out = get_cube(rgb_data=file_path_rgb, multi_band_data=file_path_multi,
                   band_order=channel_order, interpolation=interpolation_method,
                   get_tif=get_tif_file, georeference_image=georeferences,
                   file_name=ID, get_picture=get_pic)

    print('Script', __name__, 'finished')
else:
    print('Module', __name__, 'successfully imported')
