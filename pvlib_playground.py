import os
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3


class PvLibPlayground:

    cam = 1

    def __init__(self):
        pass

    @staticmethod
    def set_cam(cam):
        PvLibPlayground.cam = cam
        # print('CAM SET TO: ' + str(cam))


    @staticmethod
    def get_longitude():
        if PvLibPlayground.cam == 1:
            return 37.091549
        return 37.095253

    @staticmethod
    def get_latitude():
        if PvLibPlayground.cam == 1:
            return -2.363556
        return -2.354785

    @staticmethod
    def get_altitude():
        return 545.9

    @staticmethod
    #todo get real attitude on
    def get_location_almeria():
        return Location(PvLibPlayground.get_longitude(), PvLibPlayground.get_latitude(), tz='UTC', altitude=PvLibPlayground.get_altitude())

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
    def get_sun_earth_distance(times):
        return pvlib.solarposition.pyephem_earthsun_distance(times).values.tolist()

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
    def get_ephemeris_data(times):
        #apparent_elevation : apparent sun elevation accounting for atmospheric refraction.
        #elevation : actual elevation (not accounting for refraction) of the sun in decimal degrees, 0 = on horizon. The complement of the zenith angle.
        #apparent_zenith : apparent sun zenith accounting for atmospheric refraction.
        #solar_time : Solar time in decimal hours (solar noon is 12.00).
        data = pvlib.solarposition.ephemeris(times, PvLibPlayground.get_latitude(), PvLibPlayground.get_longitude())[['apparent_elevation',
                                                                                                                      'elevation',
                                                                                                                      'apparent_zenith',
                                                                                                                      'solar_time']].values.tolist()
        return data

    @staticmethod
    def get_meteor_data(month, day, times):  # todo add more
        csi = PvLibPlayground.get_clear_sky_irradiance(times)
        azimuth = PvLibPlayground.get_azimuth(month, day, times)
        zenith = PvLibPlayground.get_solar_zenith_angle(month, day, times)
        sun_earth_dist = PvLibPlayground.get_sun_earth_distance(times)
        ephemeris = PvLibPlayground.get_ephemeris_data(times)
        return csi, azimuth, zenith, sun_earth_dist, ephemeris


# p = PvLibPlayground()
# times1 = PvLibPlayground.get_times(2019, 10,1, 7, 19)
# times2 = PvLibPlayground.get_times(2019, 10,1, 7, 19,offset=3)
#
# print(PvLibPlayground.get_ephemeris_data(times1))
# print(PvLibPlayground.get_ephemeris_data(times2))
# t = PvLibPlayground.get_ephemeris_data(times)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(t)
#
# print(PvLibPlayground.get_azimuth(10, 15, t))
# PvLibPlayground.get_azimuth(10,15, PvLibPlayground.get_pd_time(10, 15, 12, 00))
# p.get_clear_sky_irradiance(10,1)
