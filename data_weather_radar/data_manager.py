import os
from tqdm import tqdm
from typing import Optional, List, Union

from data_weather_radar.utils import download_from_http, get_s3_url_head, copy_to_s3, get_all_file_path_s3, \
    argwrapper, imap_unordered_bar
from data_weather_radar.convert import convert_glib2

DIR_PARENT_LOCAL = 'data'
DIR_PARENT_S3 = 'data/JMA/RA'


class WeatherRadarDataManager(object):
    def __init__(self, dir_parent_local: str=DIR_PARENT_LOCAL, dir_parent_s3:str=DIR_PARENT_S3) -> None:
        """ Handling weather radar data from downloading, converting, uploading to S3

        Args:
            dir_parent_local (str): parent directory on local
            dir_parent_s3 (str): parent directory on S3
        """
        self.dir_parent_local = dir_parent_local
        self.dir_parent_s3 = dir_parent_s3

        self.dir_parent_raw_local = os.path.join(self.dir_parent_local, 'raw_')
        self.dir_parent_converted_local = os.path.join(self.dir_parent_local, 'converted')

    def convert_glib2_s3(self, url_src: str, epsg_dst: int = 4326, overwrite: bool = True, remove_local_file: bool = False,
                         multiprocessing: bool = False) -> List[str]:
        """ Convert glib2 file on s3

        Args:
            url_src (str): url of target file
            epsg_dst (int): epsg code for reprojection
            overwrite (bool): If True, overwrite existing files
            reprojection_method (str): reprojection method of gdal_warp
            multiprocessing (bool): If True, multiprocessing can be used

        Returns:
            (List[str]) urls of converted files on S3

        """


        url_head = get_s3_url_head(multiprocessing=multiprocessing)
        assert url_src.startswith(url_head) and self.dir_parent_s3 in url_src
        dir_dst_local = os.path.join(self.dir_parent_raw_local,
                                     os.path.dirname(url_src[len(url_head) + 1 + len(self.dir_parent_s3) + 1:]))
        # download
        path_local = download_from_http(url_src=url_src, dir_dst=dir_dst_local, filename=None, overwrite=overwrite)

        # convert
        dir_dest = os.path.dirname(path_local).replace(self.dir_parent_raw_local, self.dir_parent_converted_local)

        path_netcdf, path_gtiff, path_gtiff_reproj = convert_glib2(path_src=path_local,
                                                                   dir_dst=dir_dest,
                                                                   epsg_dst=epsg_dst,
                                                                   overwrite=overwrite,
                                                                   reprojection_method='cubic')

        # upload to S3
        urls_dst_s3 = []
        for path_src_local in [path_netcdf, path_gtiff, path_gtiff_reproj]:
            path_dst_s3 = os.path.join(self.dir_parent_s3, path_src_local[len(self.dir_parent_local) + 1:])
            url_dst_s3 = copy_to_s3(path_src_local=path_src_local, path_dst_s3=path_dst_s3,
                                    remove_local_file=remove_local_file, overwrite=overwrite,
                                    multiprocessing=multiprocessing)
            urls_dst_s3.append(url_dst_s3)

        if remove_local_file:
            os.remove(path_local)

        return urls_dst_s3

    def convert_glib2_s3_directory(self, dir_src_s3: str, ext_filter: Optional[Union[str, List[str]]] = '.bin',
                                   epsg_dst: int = 4326, overwrite: bool = True, remove_local_file: bool = False,
                                   processes: int = 1) -> List[str]:
        """ Convert glib2 files under specific directory on s3

        Args:
            dir_src_s3 (str): target directory
            ext_filter (list): list of string for selecting files which are matched from end of the filenames. If None, all of the files are returned
            epsg_dst (int): epsg code for reprojection
            overwrite (bool): If True, overwrite existing files
            reprojection_method (str): reprojection method of gdal_warp
            processes (int): the number of processes

        Returns:
            (List[str]) urls of converted files on S3

        """

        assert processes > 0

        urls_src = get_all_file_path_s3(dir_parent=dir_src_s3, ext_filter=ext_filter)
        if processes == 1:
            urls_dst_s3 = []
            for url_src in tqdm(urls_src, total=len(urls_src)):
                urls_temp = self.convert_glib2_s3(url_src, epsg_dst=epsg_dst, overwrite=overwrite,
                                                  remove_local_file=remove_local_file,
                                                  multiprocessing=False)
                urls_dst_s3.extend(urls_temp)
        else:
            func_args = [(self.convert_glib2_s3, url_src, epsg_dst, overwrite, remove_local_file, True)
                         for url_src in urls_src]
            urls_src = imap_unordered_bar(argwrapper, func_args, processes, extend=True)
        return urls_src
