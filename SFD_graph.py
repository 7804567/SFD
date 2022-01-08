import matplotlib.pyplot as plt
import numpy as np

from utility import *
from utility_fun import *
from mpl_toolkits.basemap import Basemap
plt.rcParams.update({'font.size':20})

class sfd_graph():

    def __init__(self, model):
        self.model = model

    def spatial_plot(self, t, weights_seas="none", weights_trend="none", code=["trendseas"],
                     normalize_mean="none", mode="display", type_plot="pcolormesh",
                     llat=-60, llon=-180, ulat=70, ulon=180, focus=False):

        if ("data" in code or "residuals" in code) and type_plot[1] != "scatter" and not self.model.data.grid:
            raise Exception("Data/residuals, can only be plot with type_plot=[, scatter].")
        if ("data" in code or "residuals" in code) and not t in self.model.data.t_data:
            raise Exception("t has to be a time point where data exists if you want to plot data or residuals")

        if not "scatter" in type_plot:

            if "data" in code or "residuals" in code:
                indice = np.where(self.model.data.t_data == t)[0][0]
                long = self.model.data.grid_p_data[0]
                lat = self.model.data.grid_p_data[1]
                latlong = np.array([lat, long])
                XYZ = cart((long.reshape(-1) + 180) * np.pi / 180, (lat.reshape(-1) + 90) * np.pi / 180)

            else:
                N = 80
                lat = np.linspace(0, np.pi, N)
                long = np.linspace(0, 2 * np.pi, 2 * N)
                latlong = np.meshgrid(lat, long)
                lat2, long2 = latlong[0].reshape(-1), latlong[1].reshape(-1)
                XYZ = cart(long2, lat2)
                long = (long - np.pi) * 180 / np.pi
                lat = (lat - np.pi / 2) * 180 / np.pi
                latlong = np.meshgrid(lat, long)

        if "scatter" in type_plot:
            XYZ = self.model.data.p_data
            long = self.model.data.p_data_SC[:, 0]
            lat = self.model.data.p_data_SC[:, 1]

            if "data" in code or "residuals" in code:
                indice = np.where(self.model.data.t_data == t)[0][0]
                indices = self.model.data.mask[indice * self.model.data.N: (indice + 1) * self.model.data.N]
                long = long[indices]
                lat = lat[indices]
                XYZ = XYZ[indices, :]

        c = []
        new_code = []
        title = []

        if normalize_mean == "seas" or normalize_mean == "trend":
            mean = self.model.mean(XYZ, np.linspace(self.model.data.t_data[0], self.model.data.t_data[-1], 1000),
                                 code=normalize_mean)

        if "trend" in code:
            if normalize_mean == "seas":
                c.append(self.model.evaluation(XYZ, t, weights_trend=weights_trend, code=["trend"]).reshape(-1)
                         + mean)
            if normalize_mean == "trend":
                c.append(self.model.evaluation(XYZ, t, weights_trend=weights_trend, code=["trend"]).reshape(-1)
                         - mean)
            else:
                c.append(self.model.evaluation(XYZ, t, weights_trend=weights_trend, code=["trend"]))
                new_code.append("trend")
                title.append("Spatial trend field estimate at time " + str(round(t, 3)) + ".")

        if "seas" in code:
            if normalize_mean == "seas":
                c.append(self.model.evaluation(XYZ, t, weights_seas=weights_seas, code=["seas"]).reshape(-1)
                         - mean)
            if normalize_mean == "trend":
                c.append(self.model.evaluation(XYZ, t, weights_seas=weights_seas, code=["seas"]).reshape(-1)
                         + mean)
            else:
                c.append(self.model.evaluation(XYZ, t, weights_seas=weights_seas, code=["seas"]))
                new_code.append("seas")
                title.append("Spatial seas field estimate at time " + str(round(t, 3)) + ".")

        if "trendseas" in code:
            c.append(self.model.evaluation(XYZ, t, weights_seas, weights_trend, code=["trend", "seas"]))
            new_code.append("trendseas")
            title.append("Spatial trendseas field estimate at time " + str(round(t, 3)) + ".")

        if "data" in code:
            if "scatter" in type_plot:
                c.append(self.model.data.data[indice * self.model.data.N: (indice + 1) * self.model.data.N][
                    indices])
            else:
                c.append(self.model.data.grid_data[indice].reshape(-1))
            new_code.append("data")
            title.append("Spatial data field at time " + str(round(t, 3)) + ".")

        if "residuals" in code:
            if "scatter" in type_plot:
                position_to_plot = self.model.data.p_data[indices, :]
                data_to_plot = self.model.data.data[indice * self.model.data.N: (indice + 1) * self.model.data.N][
                indices]
                c.append(data_to_plot.reshape(-1) -
                        self.model.evaluation(position_to_plot, t, weights_seas, weights_trend,
                                                code=["trend", "seas"]).reshape(-1))
            else:
                data_to_plot = self.model.data.grid_data[indice]
                c.append(data_to_plot.reshape(-1) -
                         self.model.evaluation(XYZ, t, weights_seas, weights_trend,
                                                code=["trend", "seas"]).reshape(-1))
            new_code.append("residuals")
            title.append("Spatial residual field at time " + str(round(t, 3)) + ".")


        if len(c) > 0:
            for i in range(len(c)):
                if not "scatter" in type_plot and not ("data" in code or "residuals" in code):
                    c[i] = c[i].reshape(2 * N, N)
                if not "scatter" in type_plot and ("data" in code or "residuals" in code):
                    c[i] = c[i].reshape(long.shape[0], long.shape[1])
                    c[i][~self.model.data.grid_mask[indice]] = 0
                if "scatter" in type_plot:
                    c[i] = c[i].reshape(-1)
            if mode == "display":
                if "scatter" in type_plot:
                    splot_scatter(long, lat, c, title, llon=llon, llat=llat, ulon=ulon, ulat=ulat, focus=focus,
                                  frame=self.model.data.frame)
                else:
                    splot_grid(latlong, c, title, llon=llon, llat=llat, ulon=ulon, ulat=ulat, focus = focus,
                                type_plot=type_plot, frame=self.model.data.frame)

    def temporal_plot(self, indice=None, spatial_position=None, code=["seas", "trend", "trendseas","data"],
                      normalize_mean="none", weights_trend="none", weights_seas="none", mode="display", geo_pos=True):

        if (indice == None and spatial_position == None) or (indice != None and spatial_position != None):
            raise Exception("You have to specify an -indice or a -spatial_position. You specified both or none.")
        if indice == None and ("data" in code or "residuals" in code):
            raise Exception("Data/residuals, can not be plot at your spatial_position as there might not be any data there."
                          " Use -indice if you want to plot data.")

        sample = np.linspace(self.model.data.t_data[0], self.model.data.t_data[self.model.data.t_data.shape[0] - 1],
                             1000)
        if indice != None:
            indices = indice + self.model.data.N * np.arange(0, self.model.data.M, 1)
            y = self.model.data.data[indices][self.model.data.mask[indices]]
            pos = [self.model.data.p_data_SC[indice, 0],  self.model.data.p_data_SC[indice, 1]]

        if indice == None:
            #lon
            pos = spatial_position

        xyz = cart((pos[0] + 180) * np.pi / 180, (pos[1] + 90) * np.pi / 180)[0]
        c = []
        new_code = []

        if geo_pos:
            fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(25, 12),
                                     gridspec_kw={
                           'width_ratios': [8, 16]})
        else:
            fig, axes = plt.subplots(ncol=1, nrows=1, figsize=(25, 12))

        if normalize_mean == "seas" or normalize_mean == "trend":
            mean = self.model.mean(xyz, np.linspace(self.model.data.t_data[0], self.model.data.t_data[-1], 1000),
                                       code=normalize_mean)

        if "trend" in code:
            if normalize_mean == "seas":
                c.append(self.model.evaluation(xyz, sample, weights_trend, code=["trend"]) + mean)
            if normalize_mean == "trend":
                c.append(self.model.evaluation(xyz, sample, weights_trend, code=["trend"]) - mean)
            else:
                c.append(self.model.evaluation(xyz, sample, weights_trend, code=["trend"]))
            new_code.append("Trend")
        if "seas" in code:
            if normalize_mean == "seas":
                c.append(self.model.evaluation(xyz, sample, weights_seas, code=["seas"]) - mean)
            if normalize_mean == "trend":
                c.append(self.model.evaluation(xyz, sample, weights_seas, code=["seas"]) + mean)
            else:
                c.append(self.model.evaluation(xyz, sample, weights_seas, code=["seas"]))
            new_code.append("Seas")
        if "trendseas" in code:
            c.append(self.model.evaluation(xyz, sample, weights_seas, weights_trend, code=["trend", "seas"]))
            new_code.append("Full: trend+seas")

        if "residuals" in code:
            res = y - self.model.evaluation(xyz, self.model.data.t_data, weights_seas, weights_trend,
                                            code=["trend", "seas"])[self.model.data.mask[indices]]


        if "data" in code and indice != None:
            if geo_pos:
                axes[1].plot(self.model.data.t_data[self.model.data.mask[indices]], y, label="linear interpolation of the data")
                axes[1].scatter(self.model.data.t_data[self.model.data.mask[indices]], y, label="data")
                axes[1].scatter(self.model.data.t_data[~self.model.data.mask[indices]],
                            np.repeat(0, self.model.data.t_data[~self.model.data.mask[indices]].shape[0]),
                            label="no data")
            else:
                axes.plot(self.model.data.t_data[self.model.data.mask[indices]], y, label="linear interpolation of the data")
                axes.scatter(self.model.data.t_data[self.model.data.mask[indices]], y, label="data")
                axes.scatter(self.model.data.t_data[~self.model.data.mask[indices]],
                            np.repeat(0, self.model.data.t_data[~self.model.data.mask[indices]].shape[0]),
                            label="no data")
        if "residuals" in code and indice!= None:
            if geo_pos:
                axes[1].scatter(self.model.data.t_data[self.model.data.mask[indices]], res, label="Residuals",
                                marker="*", color="magenta")
            else:
                axes.scatter(self.model.data.t_data[self.model.data.mask[indices]], res, label="Residuals",
                                marker="*", color="magenta")

        if geo_pos:

            map = Basemap(projection='ortho', lat_0=pos[1], lon_0=pos[0], ax=axes[0])
            map.drawmapboundary(fill_color='aqua')
            map.fillcontinents(color='coral', lake_color='aqua')
            map.drawcoastlines()
            x, y = map(pos[0], pos[1])
            map.plot(x, y, marker="D", color="k")
            if indice != None:
                axes[0].set_title("Position of the station number " + str(indice) + "\n at the position (LON:"
                                + str(round(pos[0], 2)) + ", LAT:" + str(round(pos[1], 2)) + ") on earth.")
            else:
                axes[0].set_title("Position (LON:"
                                + str(round(pos[0], 2)) + ", LAT:" + str(round(pos[1], 2)) + ") on earth.")

        for i in range(len(c)):

            if geo_pos:
                axes[1].plot(sample, c[i], label=new_code[i] + "fit of the data.")
                axes[1].set_xlabel("Time")
                axes[1].set_ylabel("Data unit")
                axes[1].set_title("Graph of different reconstructions of the signal at (LON:"
                              + str(round(pos[0], 2)) + ", LAT:" + str(round(pos[1], 2)) + ").")
                axes[1].legend()

            else:
                axes.plot(sample, c[i], label=new_code[i] + "fit of the data.")
                axes.set_xlabel("Time")
                axes.set_ylabel("Data unit")
                axes.set_title("Graph of different reconstructions of the signal at (LON:"
                              + str(round(pos[0], 2)) + ", LAT:" + str(round(pos[1], 2)) + ").")
                axes.legend()

    def rough_estimate(self, t, code=["trendseas"]):

        tilde_y = self.model.data.data
        tilde_y[~self.model.data.mask] = 0
        rough_estimate_trend = None
        rough_estimate_seas = None

        if "trend" in code or "trendseas" in code:
            rough_estimate_trend = self.model.op_trend.adjoint(tilde_y)
            rough_estimate_trend = rough_estimate_trend.reshape(self.model.M_trend, self.model.N_trend).transpose()
        if "seas" in code or "trendseas" in code:
            rough_estimate_seas = self.model.op_seas.adjoint(tilde_y)
            rough_estimate_seas = rough_estimate_seas.reshape(self.model.M_seas, self.model.N_seas).transpose()

        self.spatial_plot(t=t, weights_trend=rough_estimate_trend, weights_seas=rough_estimate_seas,
                          code=code, normalize_mean=False,llat=-90, llon=-180, ulat=90, ulon=180)

