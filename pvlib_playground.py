import os
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3
from pvlib.forecast import GFS

class PvLibPlayground:

    def __init__(self):
        pass

    @staticmethod
    def get_longitude():
        return 37.091549

    @staticmethod
    def get_latitude():
        return -2.363556

    @staticmethod
    def get_altitude():
        return 545.9

    @staticmethod
    #todo get real attitude on
    def get_location_almeria():
        return Location(37.091549, -2.363556, tz='UTC', altitude=PvLibPlayground.get_altitude())

    @staticmethod
    def get_df_times(start, end):
        tus = pd.date_range(start=start, end=end, freq='1min', tz='UTC', name=None)
        return tus

    @staticmethod
    def get_azimuth(month, day, times):
        doy = PvLibPlayground.get_day_of_year(month, day)
        declination =  pvlib.solarposition.declination_cooper69(doy)
        hour_angle = PvLibPlayground.get_hour_angle(doy, times)
        return pvlib.solarposition.solar_azimuth_analytical(PvLibPlayground.get_latitude(),
                                                            hour_angle,
                                                            declination,
                                                            pvlib.solarposition.solar_zenith_analytical(declination,
                                                                                                        hour_angle,
                                                                                                        PvLibPlayground.get_latitude()))
    @staticmethod
    def get_day_of_year(month, day):
        from datetime import datetime
        dt = datetime(year=2019, month=month, day=day)
        return dt.timetuple().tm_yday

    @staticmethod
    def get_hour_angle(day_of_year, times):
        return pvlib.solarposition.hour_angle(times,
                                              PvLibPlayground.get_longitude(),
                                              pvlib.solarposition.equation_of_time_spencer71(day_of_year))

    @staticmethod
    def get_declation_angle(month, day):
        return pvlib.solarposition.declination_cooper69(PvLibPlayground.get_day_of_year(month, day))

    def get_solar_attitude_angle(self, month, day, time):
        pass

    def get_solar_elevation_angle(self, month, day, time):
        pass

    def get_solar_incidence_angle(self, month, day, time):
        pass

    @staticmethod
    def get_solar_zenith_angle(month, day, times):
        doy = PvLibPlayground.get_day_of_year(month, day)
        return pvlib.solarposition.solar_zenith_analytical(PvLibPlayground.get_latitude(),
                                                    PvLibPlayground.get_hour_angle(doy, times),
                                                    PvLibPlayground.get_declation_angle(month, day))

    @staticmethod
    def get_pd_time(month, day, hour, minute):
        return pd.Timestamp(year=2019, month=month, day=day, hour=hour, minute=minute)

    @staticmethod
    def get_times(year, month, day, start_time, end_time, offset=0):
        s = pd.Timestamp(year=year, month=month, day=day, hour=start_time, minute=offset)
        if offset > 0:
            e = pd.Timestamp(year=year, month=month, day=day, hour=end_time, minute=offset-1)
        else:
            e = pd.Timestamp(year=year, month=month, day=day, hour=(end_time - 1), minute=59)
        return PvLibPlayground.get_df_times(s, e)


    @staticmethod
    def get_clear_sky_irradiance(times):
        loc = PvLibPlayground.get_location_almeria()
        ghic = loc.get_clearsky(times, model='ineichen', linke_turbidity=3)
        return ghic['ghi'].values.tolist()

    @staticmethod
    def get_meteor_data(month, day, times):  # todo add more
        csi = PvLibPlayground.get_clear_sky_irradiance(times)
        azimuth = PvLibPlayground.get_azimuth(month, day, times)
        zenith = PvLibPlayground.get_solar_zenith_angle(month, day, times)
        return csi, azimuth, zenith

    @staticmethod
    # expirimental
    def get_cloud_coverage(start, end):
        model = GFS()
        raw_data = model.get_data(PvLibPlayground.get_latitude(), PvLibPlayground.get_longitude(), start, end)
        data = model.process_data(raw_data)
        # data = model.get_processed_data(PvLibPlayground.get_latitude(), PvLibPlayground.get_longitude(), start, end)
        return data

# p = PvLibPlayground()
# start = pd.Timestamp(year=2020, month=1, day=23, hour=9, minute=0)
# end = pd.Timestamp(year=2020, month=1, day=23, hour=17, minute=58)
# t = p.get_cloud_coverage(start, end)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(t)
# #
# t = PvLibPlayground.get_df_times(start, end)
#
# print(PvLibPlayground.get_azimuth(10, 15, t))
# PvLibPlayground.get_azimuth(10,15, PvLibPlayground.get_pd_time(10, 15, 12, 00))
# p.get_clear_sky_irradiance(10,1)
