import os
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3


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
    def get_location_almeria():
        return Location(37.091549, -2.363556, tz='UTC', altitude=0)

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
    def get_declation_angle(self, month, day):
        return pvlib.solarposition.declination_cooper69(PvLibPlayground.get_day_of_year(month, day))

    def get_solar_attitude_angle(self, month, day, time):
        pass

    def get_solar_elevation_angle(self, month, day, time):
        pass

    def get_solar_incidence_angle(self, month, day, time):
        pass

    @staticmethod
    def get_solar_zenith_angle(self, month, day, times):
        doy = PvLibPlayground.get_day_of_year(month, day)
        return pvlib.solarposition.solar_zenith_analytical(PvLibPlayground.get_latitude(),
                                                    PvLibPlayground.get_hour_angle(doy, times),
                                                    PvLibPlayground.get_declation_angle(month, day))

    @staticmethod
    def get_pd_time(month, day, hour, minute):
        return pd.Timestamp(year=2019, month=month, day=day, hour=hour, minute=minute)

    @staticmethod
    def get_clear_sky_irradiance(month, day, start_time, end_time):
        loc = PvLibPlayground.get_location_almeria()

        start = pd.Timestamp(year=2019, month=month, day=day, hour=start_time)
        end = pd.Timestamp(year=2019, month=month, day=day, hour=(end_time-1), minute = 59)
        times = PvLibPlayground.get_df_times(start, end)
        ghic = loc.get_clearsky(times, model='ineichen', linke_turbidity=3)
        return ghic['ghi'].values.tolist()

    def get_all_external_data(self, month, day, time):
        pass


# p = PvLibPlayground()
month = 10
day = 15
start_time = 6
end_time = 18
start = pd.Timestamp(year=2019, month=month, day=day, hour=start_time)
end = pd.Timestamp(year=2019, month=month, day=day, hour=(end_time-1), minute = 59)

t = PvLibPlayground.get_df_times(start, end)

print(PvLibPlayground.get_azimuth(10, 15, t))
# PvLibPlayground.get_azimuth(10,15, PvLibPlayground.get_pd_time(10, 15, 12, 00))
# p.get_clear_sky_irradiance(10,1)
