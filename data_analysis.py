import numpy as np
import urllib.request
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import netcdf
import netCDF4

def test1234():
    temperature_original = np.zeros((27397, (2010-1961+1) * 12))
    mask_original = np.ones((27397, (2010-1961+1) * 12))

    i, j = 0, 0
    current_name, old_name = "", ""
    current_year, old_year = 2010, -1
    names = []
    with open("ghcnm.tavg.v4.0.1.20211030.qfe.dat") as file:
        for line in file:

            old_year = current_year
            old_name = current_name
            current_year = int(line[11:15])
            current_name = line[0:11]

            if current_name != old_name:
                names.append(current_name)

            if current_name == old_name and current_year != old_year + 1:
                for l in range(current_year - old_year - 1):
                    mask_original[i, j: j + 12] = 0
                    j += 12

            if current_name != old_name and (old_year != 2010 or current_year != 1961):
                #print("Issue type Boundary with station (" + str(i) + ") " + old_name)
                if old_year != 2010:
                    for l in range(2010 - old_year):
                        mask_original[i, j: j + 12] = 0
                        j += 12
                    i += 1
                    j = 0
                print("Station " + str(i) + "/27667, " + old_name + ", done.")
                if current_year != 1961:
                    for l in range(current_year - 1961):
                        mask_original[i, j: j + 12] = 0
                        j += 12

            for k in range(12):
                temp = float(line[19 + (k * 8): 19 + (k * 8) + 5]) / 100
                if temp != -99.99:
                    temperature_original[i, j + k] = temp
                else:
                    mask_original[i, j + k] = 0
            j += 12
            if j >= (2010 - 1961 + 1) * 12:
                j = 0
                i += 1
                print("Station " + str(i) + "/27667, " + current_name + ", done.")

    position = 0
    latlong = np.zeros((27397, 2))
    with open("ghcnm.tavg.v4.0.1.20211030.qfe.inv") as file:
        for line in file:
            if line[0:11] == names[position]:
                latlong[position, 0] = float(line[12:21])
                latlong[position, 1] = float(line[21:31])
                position += 1

    for i in range(600):
        year = int((i - (i % 12)) / 12)
        month = i % 12 + 1
        year += 1961
        np.save("temp_vectors/vec_" + str(year) + "_" + str(month) + ".npy", temperature_original[:, i])
        np.save("temp_vectors/mask" + str(year) + "_" + str(month) + ".npy", mask_original[:, i])

    np.save("latlong", latlong)

def load(year_start, month_start, year_end, month_end):

    temperatures = np.zeros(27397)
    mask = np.zeros(27397)
    for i in range((year_end - year_start) * 12 + month_end - (month_start - 1)):
        ref = (year_start) * 12 + month_start + i - 1
        print("year " + str(int((ref - (ref % 12)) / 12)) + " month " + str(ref % 12 + 1))
        temperatures = np.vstack((temperatures,
                                  np.load("temp_vectors/vec_" + str(int((ref - (ref % 12)) / 12)) + "_" + str(
                                      ref % 12 + 1) + ".npy")))
        mask = np.vstack((mask,
                                  np.load("temp_vectors/mask" + str(int((ref - (ref % 12)) / 12)) + "_" + str(
                                      ref % 12 + 1) + ".npy")))
    temperatures = temperatures.transpose()
    temperatures = temperatures[:, 1:]
    mask = mask.transpose()
    mask = mask[:, 1:]
    latlong = np.load("latlong.npy")

    return latlong, temperatures, mask



def save_errst():
    for year in range(168):
        actual_year = 1854 + year
        content = urllib.request.urlopen('https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/ascii/ersst.v5.'
                                         + str(actual_year) + ".asc")
        read_content = content.read()
        mat = read_content.decode("utf-8")
        mat = mat.split()

        m = np.zeros((180 * 12, 89))
        mask = np.ones((180 * 12, 89))
        i, j = 0, 0
        for t in mat:
            t_tilde = int(t) / 100
            if t_tilde != -99.99:
                m[i, j] = t_tilde
            else:
                mask[i, j] = 0
            j+=1
            if j >= 89:
                j=0
                i+=1

        for month in range(12):
            actual_month = month + 1
            np.save("errst_temp/"+str(actual_year)+"_"+str(actual_month), m[month * 180: (month + 1) * 180, :])
            np.save("errst_mask/"+str(actual_year)+"_"+str(actual_month), mask[month * 180: (month + 1) * 180, :])
        print("Year " + str(actual_year) + " saved.")

def load_errst(year_start, month_start, year_end, month_end):

    temperatures = np.zeros(180 * 89)
    mask = np.zeros(180 * 89)

    for i in range((year_end - year_start) * 12 + month_end - (month_start - 1)):
        ref = (year_start) * 12 + month_start + i - 1
        print("year " + str(int((ref - (ref % 12)) / 12)) + " month " + str(ref % 12 + 1))
        temperatures = np.vstack((temperatures,
                                  np.load("errst_temp/" + str(int((ref - (ref % 12)) / 12)) + "_" + str(
                                      ref % 12 + 1) + ".npy").reshape(-1)))
        mask = np.vstack((mask,
                                  np.load("errst_mask/" + str(int((ref - (ref % 12)) / 12)) + "_" + str(
                                      ref % 12 + 1) + ".npy").reshape(-1)))
    temperatures = temperatures.transpose()
    temperatures = temperatures[:, 1:]
    mask = mask.transpose()
    mask = mask[:, 1:]
    lat = np.arange(-88, 90, 2)
    long = np.arange(0, 360, 2)
    latlong = np.meshgrid(lat, long)
    latlong = np.vstack((latlong[0].reshape(-1), latlong[1].reshape(-1))).transpose()

    return latlong, temperatures, mask


data = netCDF4.Dataset("air.mon.anom.nc", format="NETCDF4")


