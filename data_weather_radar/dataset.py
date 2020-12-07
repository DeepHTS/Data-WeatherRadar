import os
import datetime
import pytz
from typing import List, Optional, Tuple, Union, Callable, Dict
import numpy as np
from pathlib import Path
from tqdm import tqdm

import rasterio
from rasterio.windows import Window, transform
import pandas as pd

from data_weather_radar.utils import get_all_file_path_s3, copy_to_s3, check_file_existence_local, argwrapper, \
    imap_unordered_bar


# from data_weather_radar.utils import get_all_file_path_s3


# get array
# get convert to small geotiff
# todo: prepare image base list [path, original image path, date, lat, lon, size]
#   convert, list.append(info..), upload,   after upload list
# todo: make list with train and eval with the above list


def get_array(path_img: str, lon: float, lat: float, size: Tuple[int, int] = (256, 256), pos: str = 'center',
              band: Optional[int] = 1) -> Tuple[np.array, rasterio.profiles.Profile, Window]:
    """ Get specific sized square array from geotiff data

    Args:
        path_img (str): image path or url
        lon (float): longitude
        lat (float): latitude
        size (Tuple[int, int]): numpy array size
        pos (str): If 'center', lon and lat are center position in acquired square
        band (Optional[int]): If None, all band.

    Returns:
        (Tuple[np.array, rasterio.profiles.Profile, Window]): numpy array, rasterio profile, rasterio, window

    """
    src = rasterio.open(path_img)
    py, px = src.index(lon, lat)
    if pos == 'center':
        px = px - size[0] // 2
        py = py - size[1] // 2
    window = Window(px, py, size[1], size[0])
    out_profile = src.profile.copy()
    if band is not None:
        clip = src.read(band, window=window)
    else:
        clip = src.read(window=window)

    return clip, out_profile, window

    # print(clip.shape)
    # out_profile = src.profile.copy()
    #
    # with rasterio.open('temp/test.tif', 'w', **out_profile) as dst:
    #     dst.write(clip, window=window)
    # dst.close()


def get_cropped_gtiff(path_img: str, path_out: str, lon: float, lat: float, array_size: Tuple[int, int] = (256, 256),
                      pos: str = 'center', band: Optional[int] = 1) -> None:
    """

    Args:
        path_img (str): image path or url
        path_out (str): saved path on local
        lon (float): longitude
        lat (float): latitude
        array_size (Tuple[int, int]): numpy array size
        pos (str): If 'center', lon and lat are center position in acquired square
        band (Optional[int]): If None, all band.

    Returns:

    """
    # test
    src = rasterio.open(path_img)

    py, px = src.index(lon, lat)
    if pos == 'center':
        px = px - array_size[0] // 2
        py = py - array_size[1] // 2
    window = Window(px, py, array_size[1], array_size[0])
    out_profile = src.profile.copy()

    # temp
    transform_window = transform(window, src.transform)
    out_profile["transform"] = transform_window

    if band is not None:
        clip = src.read(band, window=window)
        out_profile.update(count=1,
                           height=clip.shape[0],
                           width=clip.shape[1])

        with rasterio.open(path_out, 'w', **out_profile) as dst:
            dst.write(clip, 1)
    else:
        clip = src.read(window=window)
        out_profile.update(height=clip.shape[1],
                           width=clip.shape[2])
        with rasterio.open(path_out, 'w', **out_profile) as dst:
            dst.write(clip)

    return


def get_datetime(path_img):
    filename = os.path.basename(path_img)
    datetime_str = filename.split('_')[4]
    return datetime.datetime.strptime(datetime_str, '%Y%m%d%H%M%S%f').replace(tzinfo=pytz.utc)


def check_filename_in_time_range(path_img, datetime_start, datetime_end):
    datetime_target = get_datetime(path_img)
    return (datetime_start <= datetime_target) and (datetime_target <= datetime_end)


# # todo: this should be moved to utils in the future
# import boto3
# from data_weather_radar.utils import get_s3_url_head
#
# S3_BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
# def get_all_file_path_s3(dir_parent: str, ext_filter: Optional[Union[str, List[str]]] = None,
#                          func_kwargs: Optional[Union[Callable, Tuple[Callable, Dict]]] = None
#                          ) -> List[str]:
#     """ Get all of files' paths under specified directory on S3
#     Args:
#         dir_parent (str): parent directory for searching paths
#         ext_filter (list): list of string for selecting files which are matched from end of the filenames. If None, all of the files are returned
#     Returns:
#         (str) list of paths
#     """
#     s3_resource = boto3.resource('s3')
#     my_bucket = s3_resource.Bucket(S3_BUCKET_NAME)
#     objects = my_bucket.objects.filter(Prefix=dir_parent)
#     url_head = get_s3_url_head()
#     if ext_filter is None:
#         path_out = [os.path.join(url_head, obj.key) for obj in objects]
#     else:
#         if type(ext_filter) != list:
#             ext_filter = list(ext_filter)
#
#         path_out = []
#         for obj in objects:
#             if func_kwargs is not None:
#                 if type(func_kwargs) == tuple and (not func_kwargs[0](obj.key, **func_kwargs[1])):
#                     continue
#                 elif type(func_kwargs) != tuple and not func_kwargs(obj.key):
#                     continue
#
#             for ext_elem in ext_filter:
#                 if obj.key.endswith(ext_elem):
#                     path_out.append(os.path.join(url_head, obj.key))
#
#         # path_out = [os.path.join(url_head, obj.key) for ext_elem in ext_filter for obj in objects if
#         #             obj.key.endswith(ext_elem)]
#     return path_out


class DatasetMaker(object):
    def __init__(self, dir_parent_src, dir_parent_dst_local, dir_parent_dst_s3, subdir_dst, src_is_s3: bool = True):
        self.dir_parent_src = dir_parent_src
        self.dir_parent_dst_local = dir_parent_dst_local
        self.dir_parent_dst_s3 = dir_parent_dst_s3
        self.subdir_dst = subdir_dst
        self.src_is_s3 = src_is_s3

        self.ext_filter = '_grib2_reproj-4326.tif'
        self.cropped_name = 'cropped'
        self.list_dict_local = []
        self.list_dict_s3 = []

    def _get_candidate_files_path(self, datetime_start: datetime.datetime, datetime_end: datetime.datetime) -> List[
        str]:
        """ filter files under self.dir_parent_src with self.ext_filter and datetime range

        Args:
            datetime_start (datetime.datetime): start datetime for filtering
            datetime_end (datetime.datetime): end datetime for filtering

        Returns:
            (List[str]): filtered file path list
        """
        if self.src_is_s3:
            kwargs_filter = {
                'datetime_start': datetime_start,
                'datetime_end': datetime_end,
            }
            files_path = get_all_file_path_s3(dir_parent=self.dir_parent_src,
                                              ext_filter=self.ext_filter,
                                              func_kwargs=(check_filename_in_time_range, kwargs_filter)
                                              )
        else:
            p = Path(self.dir_parent_src)
            files_path = []
            for path_temp in p.glob("**/*" + self.ext_filter):
                if check_filename_in_time_range(path_temp, datetime_start, datetime_end):
                    files_path.append(path_temp)

        return files_path

    def get_cropped_tiff(self, path_img: str, lon: float, lat: float,
                         array_size: Tuple[int, int] = (256, 256), pos: str = 'center', band: Optional[int] = 1,
                         overwrite: bool = True,
                         s3_upload: bool = True, remove_local_file: bool = False, multiprocessing: bool = False):

        filename = os.path.basename(path_img)
        filename = os.path.splitext(filename)[0] + '_' + self.cropped_name + os.path.splitext(filename)[1]

        path_out = os.path.join(self.dir_parent_dst_local, self.subdir_dst, filename)

        if overwrite or not check_file_existence_local(path_out):
            get_cropped_gtiff(path_img=path_img,
                              path_out=path_out,
                              lon=lon, lat=lat, array_size=array_size, pos=pos, band=band)

        if s3_upload:
            path_dst_s3 = os.path.join(self.dir_parent_dst_s3, self.subdir_dst, filename)
            url_out = copy_to_s3(path_src_local=path_out, path_dst_s3=path_dst_s3, remove_local_file=remove_local_file,
                                 overwrite=overwrite, multiprocessing=multiprocessing)
        else:
            url_out = None

        return path_out, url_out

    def _get_list_of_files(self, files_path_database, files_path_origin, support_info):
        if len(files_path_database) == 0:
            return []

        assert len(files_path_database) != len(files_path_origin), print('check length of input args')

        list_info = []
        for i, file_path_database in enumerate(files_path_database):
            file_path_origin = files_path_origin[i]
            datetime_target = get_datetime(file_path_origin)
            dict_info = {
                'path_image': files_path_database,
                'path_origin': files_path_origin,
                'datetime': datetime_target,
            }
            dict_info.update(support_info)
            list_info.append(dict_info)
        return list_info

    def prepare_dataset(self, datetime_start: datetime.datetime, datetime_end: datetime.datetime,
                        lon: float, lat: float,
                        array_size: Tuple[int, int] = (256, 256), pos: str = 'center', band: Optional[int] = 1,
                        overwrite: bool = True,
                        s3_upload: bool = True, remove_local_file: bool = False, processes: int = 1
                        ):
        # todo: subdir to self.subdir
        files_path = self._get_candidate_files_path(datetime_start=datetime_start, datetime_end=datetime_end)
        print(files_path)
        if processes == 1:
            list_path = []
            list_url = []
            for file_path in tqdm(files_path, total=len(files_path)):
                path_out, url_out = self.get_cropped_tiff(path_img=file_path,
                                                          lon=lon, lat=lat,
                                                          array_size=array_size,
                                                          pos=pos,
                                                          band=band,
                                                          overwrite=overwrite,
                                                          s3_upload=s3_upload,
                                                          remove_local_file=remove_local_file,
                                                          multiprocessing=False)
                list_path.append(path_out)
                if url_out is not None:
                    list_url.append(url_out)
        else:
            func_args = [(self.get_cropped_tiff, files_path, lon, lat, array_size, pos, band, overwrite,
                          s3_upload, remove_local_file, True) for files_path in files_path]
            list_path_url = imap_unordered_bar(argwrapper, func_args, processes, extend=False)
            df = pd.DataFrame(list_path_url)
            list_path = df[0].to_list()
            list_url = df[1].to_list()
            list_url = [url for url in list_url if url is not None]

        dict_meta = {
            'longitude': lon,
            'latitude': lat,
            'position': pos,
            'array_size': array_size,
            'band': band
        }

        list_path_info = self._get_list_of_files(files_path_database=list_path,
                                                 files_path_origin=files_path,
                                                 support_info=dict_meta
                                                 )
        if len(list_path_info) > 0:
            self.list_dict_local.append(list_path_info)

        list_url_info = self._get_list_of_files(files_path_database=list_url,
                                                files_path_origin=files_path,
                                                support_info=dict_meta
                                                )
        if len(list_url_info) > 0:
            self.list_dict_s3.append(list_url_info)

        return list_path, list_url

    # todo: def save dict or df

    #
    # def xxx(self, date, upload s3:):
    #     pass
    #
    # def xxx(self, dateset_start, dateset_end, windos, processes):
    #     pass


# make tifff (geotiff) in local with list of filepath
# https://github.com/HansBambel/SmaAt-UNet/blob/master/utils/dataset_precip.py


if __name__ == '__main__':
    dict_meta = {
        'longitude': 1,
        'latitude': 2,
        'position': 3,
        'array_size': 4,
        'band': 5
    }

    # dir_parent_src = 'data/JMA/RA/converted/RA2016'
    # kwargs = {
    #     'datetime_start': datetime.datetime(year=2016, month=1, day=1, tzinfo=pytz.utc),
    #     'datetime_end': datetime.datetime(year=2016, month=1, day=3, tzinfo=pytz.utc),
    # }
    #
    # path_out =get_all_file_path_s3(dir_parent=dir_parent_src, ext_filter='_grib2_reproj-4326.tif',
    #                      func_kwargs=(check_filename_in_time_range, kwargs))
    # print(path_out)

    # def get_all_file_path_s3(dir_parent: str, ext_filter: Optional[Union[str, List[str]]] = None,
    #                          func_kwargs: Optional[Union[Callable, Tuple[Callable, Dict]]] = None
    #                          ) -> List[str]:
    #
    #
    #
    # path = 'temp/conv/Z__C_RJTD_20190101000000_SRF_GPV_Ggis1km_Prr60lv_ANAL_grib2_reproj-4326.tif'
    # lon = 127.8
    # lat = 26.3
    # get_array(path_img=path, lon=lon, lat=lat, size=(256, 256))
    #
    dir_parent_src = 'data/JMA/RA/converted/RA2015'
    path_local = 'temp/conv/Z__C_RJTD_20190101000000_SRF_GPV_Ggis1km_Prr60lv_ANAL_grib2_reproj-4326.tif'
    lon = 127.8
    lat = 26.3
    size = (256, 256)

    datetime_start = datetime.datetime(year=2015, month=8, day=1, hour=0, minute=0, tzinfo=pytz.utc)
    datetime_end = datetime.datetime(year=2015, month=8, day=1, hour=7, minute=59, tzinfo=pytz.utc)
    dir_parent_dst_local = 'dataset'
    dir_parent_dst_s3 = 'check_data/RA_dataset'
    src_s3 = True

    dataset_maker = DatasetMaker(dir_parent_src=dir_parent_src, dir_parent_dst_local=dir_parent_dst_local,
                                 dir_parent_dst_s3=dir_parent_dst_s3, src_s3=src_s3)
    # dataset_maker.get_cropped_tiff(path_img=path_local,
    #                                subdir_dst='temp',
    #                                lon=lon, lat=lat,
    #                                array_size=size,
    #                                pos='center',
    #                                band=1,
    #                                overwrite=True,
    #                                s3_upload=True,
    #                                remove_local_file=False,
    #                                multiprocessing=False
    #                                )
    list_path, list_url = dataset_maker.prepare_dataset(datetime_start=datetime_start,
                                                        datetime_end=datetime_end,
                                                        subdir_dst='temp',
                                                        lon=lon, lat=lat,
                                                        array_size=size,
                                                        pos='center',
                                                        band=1,
                                                        overwrite=True,
                                                        s3_upload=True,  # s3_upload=True
                                                        remove_local_file=False,
                                                        processes=10)
    print(list_path, list_url)
