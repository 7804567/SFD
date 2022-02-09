import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import os
import tqdm
from utility import *
from utility_fun import *
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
plt.rcParams.update({'font.size':20})

class sfd_graph():

    def __init__(self, model):
        self.model = model

    def spatial_plot(self, t, weights_seas="none", weights_trend="none", code=["trendseas"],
                     normalize_mean="none", display=True, type_plot="pcolormesh",
                     llat=-60, llon=-180, ulat=70, ulon=180, focus=False, vmin=None, vmax=None,
                     cmap=None, N=80):

        if ("data" in code or "residuals" in code) and type_plot[1] != "scatter" and not self.model.data.grid:
            raise Exception("Data/residuals, can only be plot with type_plot=[, scatter].")
        if ("data" in code or "residuals" in code) and not t in self.model.data.t_data:
            raise Exception("t has to be a time point where data exists if you want to plot data or residuals")

        if not "scatter" in type_plot:

            if "data" in code or "residuals" in code:
                indice = np.where(self.model.data.t_data == t)[0][0]
                indices = self.model.data.mask[indice]
                long = self.model.data.grid_p_data[0]
                lat = self.model.data.grid_p_data[1]
                latlong = np.array([lat, long])
                XYZ = cart((long.reshape(-1) + 180) * np.pi / 180, (lat.reshape(-1) + 90) * np.pi / 180)

            else:
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

        if type(normalize_mean) == str and (normalize_mean == "seas" or normalize_mean == "trend"):
            mean = self.model.mean(XYZ, code=normalize_mean)
        if type(normalize_mean) == list and (normalize_mean[0] == "seas" or normalize_mean[0] == "trend"):
            mean = normalize_mean[1]
            normalize_mean = normalize_mean[0]

        if "trend" in code:
            if normalize_mean == "seas":
                c.append(self.model.evaluation(XYZ, t, weights_trend=weights_trend, code=["trend"]).reshape(-1)
                         + mean)
            elif normalize_mean == "trend":
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
            elif normalize_mean == "trend":
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
                c.append(self.model.data.grid_data[indice].reshape(-1) -
                         self.model.evaluation(XYZ, t, weights_seas, weights_trend, code=["trend", "seas"]).reshape(-1))
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

            if display:
                if "scatter" in type_plot:
                    splot_scatter(long, lat, c, title, llon=llon, llat=llat, ulon=ulon, ulat=ulat, focus=focus,
                                    frame=self.model.data.frame)
                else:
                    splot_grid(latlong, c, title, llon=llon, llat=llat, ulon=ulon, ulat=ulat, focus=focus,
                                type_plot=type_plot, frame=self.model.data.frame, vmin=vmin, vmax=vmax, cmap=cmap,
                                display=display)

            else:
                if "scatter" in type_plot:
                    return splot_scatter(long, lat, c, title, llon=llon, llat=llat, ulon=ulon, ulat=ulat, focus=focus,
                                    frame=self.model.data.frame)
                else:
                    return splot_grid(latlong, c, title, llon=llon, llat=llat, ulon=ulon, ulat=ulat, focus = focus,
                                type_plot=type_plot, frame=self.model.data.frame, vmin=vmin, vmax=vmax, cmap=cmap,
                                display=display)

    def temporal_plot(self, indice=None, spatial_position=None, code=["seas", "trend", "trendseas","data"],
                      normalize_mean="none", weights_trend="none", weights_seas="none", mode="display", geo_pos=True,
                      linewidth=[]):

        if (indice == None and spatial_position == None) or (indice != None and spatial_position != None):
            raise Exception("You have to specify an -indice or a -spatial_position. You specified both or none.")
        if indice == None and ("data" in code or "residuals" in code):
            raise Exception("Data/residuals, can not be plot at your spatial_position as there might not be any data there."
                          " Use -indice if you want to plot data.")

        sample = np.linspace(self.model.data.t_data[0], self.model.data.t_data[self.model.data.t_data.shape[0] - 1],
                             100 * np.abs(int((self.model.data.t_data[-1] - self.model.data.t_data[0]) / self.model.T)))
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
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(25, 12))


        if normalize_mean == "seas" or normalize_mean == "trend":
            mean = self.model.mean(xyz, code=normalize_mean)

        if "trend" in code:
            if normalize_mean == "seas":
                c.append(self.model.evaluation(xyz, sample, weights_trend, code=["trend"]) + mean)
            elif normalize_mean == "trend":
                c.append(self.model.evaluation(xyz, sample, weights_trend, code=["trend"]) - mean)
            else:
                c.append(self.model.evaluation(xyz, sample, weights_trend, code=["trend"]))
            new_code.append("Trend")
        if "seas" in code:
            if normalize_mean == "seas":
                c.append(self.model.evaluation(xyz, sample, weights_seas, code=["seas"]) - mean)
            elif normalize_mean == "trend":
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

        if len(linewidth) == 0 or len(linewidth) != len(code):
            linewidth = [1] * len(code)

        if "data" in code and indice != None:
            index = code.index("data")
            if geo_pos:
                axes[1].plot(self.model.data.t_data[self.model.data.mask[indices]], y, label="linear interpolation of the data",
                             linewidth=linewidth[index])
                axes[1].scatter(self.model.data.t_data[self.model.data.mask[indices]], y, label="data")
                axes[1].scatter(self.model.data.t_data[~self.model.data.mask[indices]],
                            np.repeat(0, self.model.data.t_data[~self.model.data.mask[indices]].shape[0]),
                            label="no data")
            else:
                axes.plot(self.model.data.t_data[self.model.data.mask[indices]], y, label="linear interpolation of the data",
                          linewidth=linewidth[index])
                axes.scatter(self.model.data.t_data[self.model.data.mask[indices]], y, label="data")
                axes.scatter(self.model.data.t_data[~self.model.data.mask[indices]],
                            np.repeat(0, self.model.data.t_data[~self.model.data.mask[indices]].shape[0]),
                            label="no data")
            del linewidth[index]

        if "residuals" in code and indice!= None:
            index = code.index("residuals")
            if geo_pos:
                axes[1].scatter(self.model.data.t_data[self.model.data.mask[indices]], res, label="Residuals",
                                marker="*", color="magenta", s=linewidth[index])
            else:
                axes.scatter(self.model.data.t_data[self.model.data.mask[indices]], res, label="Residuals",
                                marker="*", color="magenta", s=linewidth[index])
            del linewidth[index]

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
                axes[1].plot(sample, c[i], label=new_code[i] + "fit of the data.", linewidth=linewidth[i])
                axes[1].set_xlabel("Time")
                axes[1].set_ylabel("Data unit")
                axes[1].set_title("Graph of different reconstructions of the signal at (LON:"
                              + str(round(pos[0], 2)) + ", LAT:" + str(round(pos[1], 2)) + ").")
                axes[1].legend()

            else:
                axes.plot(sample, c[i], label=new_code[i] + "fit of the data.", linewidth=linewidth[i])
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

    def save_frames(self, time_limit, dt, folder_name, code, focus=False, normalize_mean="none", M=120):

        path = os.path.join(os.getcwd(), folder_name)
        N = int(time_limit / dt)

        if not (os.path.exists(path) and os.path.isdir(path)):
            os.makedirs(path)
        pattern = os.path.join(path, "frame_*.png")
        os.system(f"rm {pattern}")
        print(f"\nSaving the frames of the animation in folder {path}")

        vmin = np.min(self.model.data.grid_data)
        vmax = np.max(self.model.data.grid_data)
        cmap = plt.get_cmap('jet', 2000)
        times = np.linspace(0, time_limit, N)

        #computation of the mean if normalization
        if normalize_mean == "seas" or normalize_mean == "trend":
            lat = np.linspace(0, np.pi, M)
            long = np.linspace(0, 2 * np.pi, 2 * M)
            latlong = np.meshgrid(lat, long)
            lat2, long2 = latlong[0].reshape(-1), latlong[1].reshape(-1)
            XYZ = cart(long2, lat2)
            mean = self.model.mean(XYZ, code=normalize_mean)
            normalize_mean = [normalize_mean, mean]

        #computation of vmin/vmax for the seas normalization
        if (code == "seas" or "seas" in code) and normalize_mean[0] == "seas":

            #t = np.linspace(0, self.model.T, 1000)
            #ev = self.model.field_phi_seas.evaluate(self.model.field_phi_seas.proj(np.clip(proj_circle(self.model.T, t).
             #                                       dot(self.model.field_phi_seas.nodes.transpose()), -1, 1)))
            #mean = self.model.field_psi_seas.evaluate(self.model.field_psi_seas.proj(np.clip(XYZ.dot(self.model.
             #       field_psi_seas.nodes.transpose()), -1, 1))).dot(self.model.estimate_seas).dot(np.mean(ev, axis=0))

            ev_normalized = self.model.evaluation(xyz=XYZ, t=np.linspace(0, self.model.T, int(self.model.T) * 20), code="seas") - \
                    mean.reshape(XYZ.shape[0], 1) * np.ones(int(self.model.T) * 20).reshape(1, int(self.model.T) * 20)

            vmin = np.min(ev_normalized)
            vmax = np.max(ev_normalized)

        #computation of vmin/vmax for the trend normalization
        if (code == "trend" or "trend" in code) and normalize_mean[0] == "trend":

            bachs_size = 100
            n_bachs = int(N / bachs_size)
            vmax = -np.inf
            vmin = np.inf
            eval = self.model.field_psi_trend.evaluate(self.model.field_psi_trend.proj(np.clip(
                XYZ.dot(self.model.field_psi_trend.nodes.transpose()), -1, 1))) \
                .dot(self.model.estimate_trend)

            for i in range(n_bachs):

                eval_normalized = eval.dot(self.model.field_zeta_trend.evaluate(self.model.field_zeta_trend.proj(
                                self.model.field_zeta_trend.nodes.reshape(self.model.M_trend, 1) - times[i * bachs_size:
                                (i + 1) * bachs_size].reshape(1, times[i * bachs_size: (i + 1) * bachs_size].shape[0]))))
                eval_normalized -= mean.reshape(XYZ.shape[0], 1) * np.ones(bachs_size).reshape(1, bachs_size)
                vmin = min(vmin, np.min(eval_normalized))
                vmax = max(vmax, np.max(eval_normalized))

                print("vmin, vmax computed at " + str(round((i + 1) * bachs_size * 100 / N, 3)) + " %.")

            if N - int(bachs_size * n_bachs) > 0:
                eval_normalized = eval.dot(self.model.field_zeta_trend.evaluate(self.model.field_zeta_trend.proj(
                     self.model.field_zeta_trend.nodes.reshape(self.model.M_trend, 1) -
                    times[bachs_size * n_bachs:].reshape(1, times[bachs_size * n_bachs:].shape[0]))))
                eval_normalized -= mean.reshape(XYZ.shape[0], 1) * np.ones(eval_normalized.shape[1])\
                                    .reshape(1, eval_normalized.shape[1])
                vmin = min(vmin, np.min(eval_normalized))
                vmax = max(vmax, np.max(eval_normalized))

        #computation of the frames if no structure
        if "residuals" in code or "data" in code:

            for i in range(N):

                figaxes = self.spatial_plot(t=times[i], code=code, type_plot="pcolormesh", focus=focus, display=False,
                                        vmin=vmin, vmax=vmax, cmap=cmap, normalize_mean=normalize_mean, N=M)

                fig = figaxes[0]
                axes = figaxes[1]
                year = (int(times[i]) - (int(times[i]) % 12)) / 12
                month = int(times[i]) % 12
                day = round((times[i] - int(times[i])) * 30, 3)
                axes.set_title("Year " + "{:11.0f}".format(year) + "; Month " + "{:7.0f}".format(month)\
                + "; Day " + "{:7.3f}".format(day) + ".")
                fig.savefig(os.path.join(path, f"frame_{i}.png"))
                plt.close()
                print("Frame " + str(i + 1) + "/" + str(N) + " saved.")

        #computation of the frames if structure
        if "seas" in code or "trend" in code or "trendseas" in code:

            bachs_size = 100
            bachs_number = int(N / bachs_size)

            lat = np.linspace(0, np.pi, M)
            long = np.linspace(0, 2 * np.pi, 2 * M)
            latlong = np.meshgrid(lat, long)
            lat2, long2 = latlong[0].reshape(-1), latlong[1].reshape(-1)
            XYZ = cart(long2, lat2)
            long = (long - np.pi) * 180 / np.pi
            lat = (lat - np.pi / 2) * 180 / np.pi
            latlong = np.meshgrid(lat, long)

            if "trend" in code or "trendseas" in code:
                eval_spatial_trend = self.model.field_psi_trend.evaluate(self.model.field_psi_trend.proj(np.clip(
                    XYZ.dot(self.model.field_psi_trend.nodes.transpose()), -1, 1))).dot(self.model.estimate_trend)

            if "seas" in code or "trendseas" in code:
                eval_spatial_seas = self.model.field_psi_seas.evaluate(self.model.field_psi_seas.proj(np.clip(
                    XYZ.dot(self.model.field_psi_seas.nodes.transpose()), -1, 1))).dot(self.model.estimate_seas)

            for i in range(bachs_number):

                if "trend" in code or "trendseas" in code:
                    eval_trend = eval_spatial_trend.dot(self.model.field_zeta_trend.evaluate(self.model.field_zeta_trend.proj(
                                self.model.field_zeta_trend.nodes.reshape(self.model.M_trend, 1) - times[i * bachs_size:
                                (i + 1) * bachs_size].reshape(1, times[i * bachs_size: (i + 1) * bachs_size].shape[0]))))
                    if normalize_mean[0] == "seas":
                        eval_trend += mean.reshape(XYZ.shape[0], 1) * np.ones(bachs_size).reshape(1, bachs_size)
                    if normalize_mean[0] == "trend":
                        eval_trend -= mean.reshape(XYZ.shape[0], 1) * np.ones(bachs_size).reshape(1, bachs_size)

                if "seas" in code or "trendseas" in code:
                    eval_seas = eval_spatial_seas.dot(self.model.field_phi_seas.evaluate(self.model.field_phi_seas.proj(
                                                np.clip(self.model.field_phi_seas.nodes.dot(proj_circle(self.model.T,
                                                times[i * bachs_size: (i + 1) * bachs_size]).transpose()), -1, 1))))
                    if normalize_mean[0] == "seas":
                        eval_seas -= mean.reshape(XYZ.shape[0], 1) * np.ones(bachs_size).reshape(1, bachs_size)
                    if normalize_mean[0] == "trend":
                        eval_seas += mean.reshape(XYZ.shape[0], 1) * np.ones(bachs_size).reshape(1, bachs_size)

                if "trendseas" in code:
                    eval = eval_trend + eval_seas
                if "seas" in code:
                    eval = eval_seas
                if "trend" in code:
                    eval = eval_trend

                for j in range(bachs_size):

                    figaxes = splot_grid(latlong, eval[:, j].reshape(2 * M, M), "", llat=-60, llon=-180, ulat=70, ulon=180, focus=focus,
                              type_plot="pcolormesh", frame=self.model.data.frame, vmin=vmin, vmax=vmax, cmap=cmap,
                              display=False)
                    fig = figaxes[0]
                    axes = figaxes[1]
                    indx = (i * bachs_size) + j
                    year = (int(times[indx]) - (int(times[indx]) % 12)) / 12
                    month = int(times[indx]) % 12
                    day = round((times[indx] - int(times[indx])) * 30, 3)
                    axes.set_title("Year " + "{:11.0f}".format(year) + "; Month " + "{:7.0f}".format(month)\
                    + "; Day " + "{:7.3f}".format(day) + ".")
                    fig.savefig(os.path.join(path, f"frame_{indx}.png"))
                    plt.close()
                    print("Frame " + str(indx + 1) + "/" + str(N) + " saved.")

            if bachs_size * bachs_number < N:

                if "trend" in code or "trendseas" in code:
                    eval_trend = eval_spatial_trend.dot(self.model.field_zeta_trend.evaluate(self.model.field_zeta_trend.proj(
                                self.model.field_zeta_trend.nodes.reshape(self.model.M_trend, 1) -
                                times[bachs_number * bachs_size:].reshape(1, times[bachs_number * bachs_size:].shape[0]))))
                    if normalize_mean[0] == "seas":
                        eval_trend += mean.reshape(XYZ.shape[0], 1) * \
                                      np.ones(eval_trend.shape[1]).reshape(1, eval_trend.shape[1])
                    if normalize_mean[0] == "trend":
                        eval_trend -= mean.reshape(XYZ.shape[0], 1) * \
                                      np.ones(eval_trend.shape[1]).reshape(1, eval_trend.shape[1])

                if "seas" in code or "trendseas" in code:
                    eval_seas = eval_spatial_seas.dot(self.model.estimate_seas).dot(self.model.field_phi_seas.evaluate(
                        self.model.field_phi_seas.proj(np.clip(self.model.field_phi_seas.nodes
                        .dot(proj_circle(self.model.T, times[bachs_number * bachs_size:]).transpose()), -1, 1))))
                    if normalize_mean[0] == "seas":
                        eval_trend += mean.reshape(XYZ.shape[0], 1) * \
                                      np.ones(eval_seas.shape[1]).reshape(1, eval_seas.shape[1])
                    if normalize_mean[0] == "trend":
                        eval_trend -= mean.reshape(XYZ.shape[0], 1) * \
                                      np.ones(eval_seas.shape[1]).reshape(1, eval_seas.shape[1])

                if "trendseas" in code:
                    eval = eval_trend + eval_seas
                if "seas" in code:
                    eval = eval_seas
                if "trend" in code:
                    eval = eval_trend

                for j in range(eval.shape[1]):

                    figaxes = splot_grid(latlong, eval[:, j], "", llat=-60, llon=-180, ulat=70, ulon=180, focus=focus,
                              type_plot="pcolormesh", frame=self.model.data.frame, vmin=vmin, vmax=vmax, cmap=cmap,
                              display=False)
                    fig = figaxes[0]
                    axes = figaxes[1]
                    indx = (bachs_number * bachs_size) + j
                    year = (int(times[indx]) - (int(times[indx]) % 12)) / 12
                    month = int(times[indx]) % 12
                    day = round((times[indx] - int(times[indx])) * 30, 3)
                    axes.set_title("Year " + "{:11.0f}".format(year) + "; Month " + "{:7.0f}".format(month)\
                    + "; Day " + "{:7.3f}".format(day) + ".")
                    fig.savefig(os.path.join(path, f"frame_{indx}.png"))
                    plt.close()
                    print("Frame " + str(indx + 1) + "/" + str(N) + " saved.")

        pattern = os.path.join(path, "frame_%01d.png")
        target = os.path.join(path, "Animation.mp4")
        print("All the frames have been generated. You can merge them in a video using ffmpeg with the following command:")
        command = f"ffmpeg -framerate {int(1 / dt)} -i {pattern} -y -c:v libx264 -c:a aac -strict experimental -tune fastdecode -pix_fmt yuv420p {target}"
        os.system(command)


