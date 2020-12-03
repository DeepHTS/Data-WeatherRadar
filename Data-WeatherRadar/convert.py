import os
import subprocess
from osgeo import gdal
from typing import Tuple, Optional


def convert_glib2(path_src: str, dir_dst: Optional[str] = None, epsg_dst: int = 4326, overwrite: bool = True,
                  reprojection_method: str = 'cubic') -> Tuple[str, str, str]:
    """ Convert glib2 file to netCDF, geotiff, and re-projected geotiff

    Args:
        path_src (str): path of target file
        dir_dst (Optional[str]): destination directory
        epsg_dst (int): epsg code for reprojection
        overwrite (bool): If True, overwrite existing files
        reprojection_method (str): reprojection method of gdal_warp

    Returns:
        (Tuple[str, str, str]) path of netCDF, geotiff, and re-projected geotiff

    """
    # convert to netCDF
    filename_src = os.path.basename(path_src)
    dir_src = os.path.dirname(path_src)

    if dir_dst is None:
        dir_dst = str(dir_src)

    if not os.path.exists(dir_dst):
        os.makedirs(dir_dst)

    filename_netcdf = filename_src.replace('.bin', '.nc')
    path_netcdf = os.path.join(dir_dst, filename_netcdf)

    if not overwrite and os.path.exists(path_netcdf) and os.path.getsize(path_netcdf) > 0:
        print('not overwrite: ', path_netcdf)
    else:
        cmd_netcdf = "wgrib2 %s -netcdf %s" % (path_src, path_netcdf)
        subprocess.call(cmd_netcdf, shell=True)

    # convert to gtiff
    path_gtiff = path_netcdf.replace('.nc', '.tif')

    if not overwrite and os.path.exists(path_gtiff) and os.path.getsize(path_gtiff) > 0:
        print('not overwrite: ', path_gtiff)
    else:
        cmd_gtiff = "gdal_translate -of GTiff -sds %s %s" % (path_netcdf, path_gtiff)
        subprocess.call(cmd_gtiff, shell=True)

    # re-project gtiff
    path_gtiff_reproj = path_gtiff.replace('.tif', '_reproj-{}.tif'.format(str(epsg_dst)))

    if not overwrite and os.path.exists(path_gtiff_reproj) and os.path.getsize(path_gtiff_reproj) > 0:
        print('not overwrite: ', path_gtiff_reproj)
    else:
        cmd_reproj = "gdalwarp -overwrite -of GTIFF -r {0} -t_srs EPSG:{1} {2} {3}".format(reprojection_method,
                                                                                           epsg_dst,
                                                                                           path_gtiff,
                                                                                           path_gtiff_reproj)
        subprocess.call(cmd_reproj, shell=True)

    return path_netcdf, path_gtiff, path_gtiff_reproj


if __name__ == '__main__':
    path_glib2 = 'temp/Z__C_RJTD_20190101000000_SRF_GPV_Ggis1km_Prr60lv_ANAL_grib2.bin'
    convert_glib2(path_glib2, dir_dst='temp/conv')
