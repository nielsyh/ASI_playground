import numpy as np
import matplotlib.pyplot as plt
import math

class swingingDoor:

    def __init__(self):
        pass

    def init_Snap(self, archived_pnt, value, trade_date, time, POSITIVE_DEV, NEGATIVE_DEV):
        prev_val = float(archived_pnt['value'])
        prev_time = int(archived_pnt['time_value'])
        time = int(time)
        value = float(value)
        Smax = (value+POSITIVE_DEV*value-prev_val)/(time-prev_time)
        Smin = (value-NEGATIVE_DEV*value-prev_val)/(time-prev_time)
        slope = (value-prev_val)/(time-prev_time)

        return {
            'value' : value,
            'trade_date' : trade_date,
            'time': time,
            'Smax': Smax,
            'Smin': Smin,
            'Slope' : slope
        }

    def snap2archive(self, snapshot, bool):
        return {
            'value' : snapshot['value'],
            'trade_date' : snapshot['trade_date'],
            'time_value' : snapshot['time'],
            'is_snap' : bool,
        }

    def compress(self, time_series, KWH_SENS, avg_mins): # returns SW compression of time-series
        ARCHIVE = []
        counter = 0
        archive_count = 0
        res = []
        times= []

        POSITIVE_DEV = KWH_SENS/100 #np.std(ls)/500
        NEGATIVE_DEV = POSITIVE_DEV

        for idx, val in enumerate(time_series):
            value = val
            trade_date = idx

            if counter == 0:
                # This is the header so we skip this iteration
                pass

            elif counter == 1:
                # This is the first data point, always added into archive
                ARCHIVE = [{
                    'value' : value,
                    'trade_date' : trade_date,
                    'time_value' : counter,
                    'is_snap'   : False,
                }]
                archive_count += 1

            elif counter == 2:
                # This is the first snapshot that we will recieved
                SNAPSHOT = self.init_Snap(
                    ARCHIVE[archive_count-1],
                    value,
                    trade_date,
                    counter,
                    POSITIVE_DEV,
                    NEGATIVE_DEV,
                )

                tmp_arch = self.snap2archive(SNAPSHOT, False)
                ARCHIVE.append(tmp_arch)
                res.append(tmp_arch['value'])
                times.append(tmp_arch['trade_date'])

            else:
                # Set up incoming value
                INCOMING = self.init_Snap(
                    ARCHIVE[archive_count-1],
                    value,
                    trade_date,
                    counter,
                    POSITIVE_DEV,
                    NEGATIVE_DEV,
                )
                if SNAPSHOT['Smin'] <= INCOMING['Slope'] <= SNAPSHOT['Smax']:
                    # It is within the filtration bounds, edit the INCOMING and
                    # set the SNAP. When editing INCOMING, make sure that the incoming
                    # slopes are not bigger than the current SNAPSHOT's slopes
                    INCOMING['Smax'] = min(SNAPSHOT['Smax'],INCOMING['Smax'])
                    INCOMING['Smin'] = max(SNAPSHOT['Smin'],INCOMING['Smin'])
                    SNAPSHOT = INCOMING
                else:
                    # It is outside the bounds so we must archive the current SNAP
                    # and init a new snap using this new archived point and INCOMING
                    tmp_arch = self.snap2archive(SNAPSHOT, False)
                    ARCHIVE.append(tmp_arch)
                    res.append(tmp_arch['value'])
                    times.append(tmp_arch['trade_date'])

                    archive_count += 1
                    SNAPSHOT = self.init_Snap(
                        ARCHIVE[archive_count-1],
                        value,
                        trade_date,
                        counter,
                        POSITIVE_DEV,
                        NEGATIVE_DEV,
                    )
            counter += 1
        # Always add the latest point into the archive
        tmp_arch = self.snap2archive(SNAPSHOT, True)
        ARCHIVE.append(tmp_arch)
        res.append(tmp_arch['value'])
        times.append(tmp_arch['trade_date'])

        return self.average_per_hour(res, times, avg_mins)  #average by the hour
        # return res, times

    def average_per_hour(self, series, times, minutes):
        res_times = []
        res_series = []

        end = times[-1]
        end = math.ceil(end / minutes) * minutes

        for i in range(0, end, minutes):
            min  = i
            max = i + minutes
            tmp_observations = []

            last_idx = 0
            for idx, val in enumerate(times):
                if val >= min and val <= max:
                    last_idx = idx
                    tmp_observations.append(series[idx])

            res_times.extend([x for x in range(i,i+minutes)])

            if len(tmp_observations) < 1:
                tmp_observations = [res_series[-1]]
                # tmp_observations = [res_series[-1], series[last_idx+1]]
                # print(tmp_observations)

            avg = np.average(tmp_observations)

            res_series.extend([avg for x in range(0,minutes)])


        return res_series, res_times



def calc_ramp_score(reference_x, reference_y, competing_x, competing_y, avg_mins):  #     rs = 1/n Integral |SD ts - SD ref|
    t_min = reference_x[0]
    t_max = reference_x[-1]

    RS = []
    for i in range(t_min, t_max,avg_mins):
        RS.append(abs(np.trapz(y=competing_y[i:i+avg_mins], x=competing_x[i:i+avg_mins]) - np.trapz(y=reference_y[i:i+avg_mins], x=reference_x[i:i+avg_mins])))
    return (1/(t_max - t_min)) * sum(RS)




def get_ramp_score(ref_ls, model_ls, avg_mins=5, sens = 80, name='Compete', plot=True):
    kwh_sens = sens
    kwh_sens = kwh_sens /100

    SD = swingingDoor()
    y_reference, x_reference = SD.compress(ref_ls,kwh_sens, avg_mins)
    y_compete, x_compete = SD.compress(model_ls,kwh_sens, avg_mins)

    if plot:
        plt.plot(ref_ls, linestyle='-', color='gray', label='Actual')
        plt.plot(x_reference, y_reference, color='blue', linestyle=':', label='SD Observed')
        plt.plot(x_compete, y_compete, color='red', linestyle=':', label='SD Competing model')

        fz = 20
        plt.title('SD compression ' + str(name), fontsize=fz)
        plt.xlabel('Time in minutes', fontsize=fz)
        plt.ylabel('GHI in kWh per square meter', fontsize=fz)

        plt.legend()
        plt.show()
        plt.close()

    rs = calc_ramp_score(x_reference, y_reference, x_compete, y_compete, avg_mins)
    # print(rs)
    return rs

# from data.data_helper import get_thesis_test_days, get_persistence_dates
#
# # t = get_thesis_test_days()
# t = get_thesis_test_days(in_cloudy=False, in_parcloudy=False, in_sunny=True)
# actual, pred, _ = get_persistence_dates(t, 6, 19, 20)
# get_ramp_score(actual, pred, avg_mins=10, sens=80)