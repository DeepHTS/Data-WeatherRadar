from data_weather_radar.data_manager import WeatherRadarDataManager

if __name__ == '__main__':
    dir_src_s3 = 'data/JMA/RA'
    weather_radar_data_manager = WeatherRadarDataManager()
    urls_dst_s3 = weather_radar_data_manager.convert_glib2_s3_directory(dir_src_s3=dir_src_s3, ext_filter='.bin',
                                                                        remove_local_file=True,
                                                                        processes=14)