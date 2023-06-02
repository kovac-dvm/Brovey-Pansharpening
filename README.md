# Pan-Sharpening Software: Enhancing Multi-Spectral Images for Geospatial Analysis

Function "get_cube" performs pan-sharpening of low-resolution data from multi-spectral camera based on the
high-resolution RGB image. If the pixel size of the data from a multi-spectral camera is not uniform compared to that
of a RGB image, up-sampling is performed. The function returns a three-dimensional ndarray of type float32 and
generates georeferenced .TIF file and optionally a .PDF file for analyzing specific bands and pan-sharpening results.

![flow_chart](https://github.com/kovac-dvm/Brovey-Pansharpening/assets/85240065/dd024764-584e-4c39-a308-4f54daa8a34e)
