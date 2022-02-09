import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from utility import *


class cdata:

    def __init__(self, mask, data, p_data, t_data, grid=False, frame=None):

        ###ABOUT THE DATA FORMAT
        #lon in (-180, 180) and lat in (-90, 90)
        ##IF grid=False
        #M and N are dimensions
        #It is supposed that t_data is a M vector with the times at which the information is situated
        #It is supposed that p_data is a Nx2 vector with the first column being the long and the second the lat
        #It is supposed that data is a NM vector with the first N entries being the data on the points p_data at time
        #t_data[0], the second N entries being the data on the points p_data at time t_data[1], etc.
        #It is supposed that mask is constructed in the same manner as data
        ##If grid=True
        #b,c, M are dimensions
        #It is supposed that t_data is a M vector with the times at which the information is situated
        #It is supposed that p_data is a list with two entries, the first entry is a bxc array with the lon,
        #the second entry is a bxc array with the lat.
        #It is supposed that data is a Mxbxc array with data[0,:,:] being the data associated to the points p_data and
        #t_data[0], etc.
        #It is supposed that mask is constructed in the same manner as data
        ###

        self.grid = grid
        #variable framed is here to take into account a possible restriction of the sphere to a section of it
        #framed should be in defined, by the user, as np.array([[llon, ulon],[llat, ulat]])
        self.frame = frame

        if self.grid == False:
            self.mask = np.array(mask, dtype=bool).transpose().reshape(-1)
            self.data = data.transpose().reshape(-1)
            self.type = type(data)
            self.p_data = np.copy(p_data)

        if self.grid == True:
            self.grid_data = data
            self.grid_mask = np.array(mask, dtype=bool)
            self.grid_p_data = np.copy(p_data)
            self.t_data = t_data
            self.p_data = np.vstack((self.grid_p_data[0].reshape(-1), self.grid_p_data[1].reshape(-1))).transpose()
            self.data = np.zeros((self.t_data.shape[0], self.grid_data[0].shape[0], self.grid_data[0].shape[1]))
            self.mask = np.zeros((self.t_data.shape[0], self.grid_data[0].shape[0], self.grid_data[0].shape[1]), dtype=bool)
            for i in range(self.t_data.shape[0]):
                self.data[i] = self.grid_data[i]
                self.mask[i] = self.grid_mask[i]
            self.data = self.data.reshape(-1)
            self.mask = self.mask.reshape(-1)
            self.type = type(data)

        self.p_data_SC = np.copy(self.p_data)
        self.p_data = np.array(self.p_data, dtype=float)
        self.p_data[:, 0] = (self.p_data[:, 0] + 180) * np.pi / 180
        self.p_data[:, 1] = (self.p_data[:, 1] + 90) * np.pi / 180
        self.p_data = cart(self.p_data[:, 0], self.p_data[:, 1])
        self.t_data = t_data
        self.N = self.p_data.shape[0]
        self.M = self.t_data.shape[0]
        self.true_dim = int(sum(self.mask))
        self.issue_indices = np.where(self.mask == False)[0]
        self.Mask = vectortomatrix(self.mask, self.true_dim)


    def temporal_plot(self, indice):
        indices = indice + self.N * np.arange(0, self.M, 1)
        y = self.data[indices][self.mask[indices]]
        plt.plot(self.t_data[self.mask[indices]],y, label="linear interpolation of the data")
        plt.scatter(self.t_data[self.mask[indices]], y, label="data")
        plt.scatter(self.t_data[~self.mask[indices]], np.repeat(0,self.t_data[~self.mask[indices]].shape[0]), label="no data")
        plt.xlabel("Time")
        plt.ylabel("Value of f(p_data[" + str(indice) + "],Time)")
        plt.legend()

    def spatial_plot(self, indice, type_plot="scatter", llon=-180, llat=-70, ulon=180, ulat=70, focus=False):

        indices = self.mask[indice * self.N: (indice + 1) * self.N]
        data_to_plot = self.data[indice * self.N: (indice + 1) * self.N][indices]

        if type_plot == "scatter":
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(25, 12))
            if focus:
                if type(self.frame) == np.ndarray:
                    map = Basemap(projection='cyl', llcrnrlon=self.frame[0, 0], llcrnrlat=self.frame[1, 0],
                                  urcrnrlon=self.frame[0, 1], urcrnrlat=self.frame[1, 1], ax=axes)
            else:
                map = Basemap(projection='cyl', llcrnrlon=llon, llcrnrlat=llat,
                              urcrnrlon=ulon, urcrnrlat=ulat, ax=axes)
            map.drawcoastlines()

            x, y = map(self.p_data_SC[:, 0][indices],  self.p_data_SC[:, 1][indices])
            jet = plt.get_cmap('jet', 2000)
            cnorm = colors.Normalize(vmin=np.min(data_to_plot), vmax=np.max(data_to_plot))
            scalarMap = cmx.ScalarMappable(norm=cnorm, cmap=jet)
            im = map.scatter(x, y, marker='D', color=scalarMap.to_rgba(data_to_plot))

            if type(self.frame) == np.ndarray:
                x_boundary, y_boundary = map(np.array([self.frame[0, 0], self.frame[0, 1],
                                                       self.frame[0, 1], self.frame[0, 0],
                                                       self.frame[0, 0]]),
                                             np.array([self.frame[1, 1], self.frame[1, 1],
                                                       self.frame[1, 0], self.frame[1, 0],
                                                       self.frame[1, 1]]))
                map.plot(x_boundary, y_boundary, color='m', linewidth=10, marker=None)

            plt.colorbar(mappable=scalarMap, ax=axes)
            axes.set_title("Data at time " + str(self.t_data[indice]) + ".")
            plt.show()


        if type_plot == "pcolormesh":

            if self.grid:
                fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(25, 12))
                if focus:
                    if type(self.frame) == np.ndarray:
                        map = Basemap(projection='cyl', llcrnrlon=self.frame[0, 0], llcrnrlat=self.frame[1, 0],
                                      urcrnrlon=self.frame[0, 1], urcrnrlat=self.frame[1, 1], ax=axes)
                else:
                    map = Basemap(projection='cyl', llcrnrlon=llon, llcrnrlat=llat,
                                  urcrnrlon=ulon, urcrnrlat=ulat, ax=axes)
                map.drawcoastlines()
                x,y = map(self.grid_p_data[0], self.grid_p_data[1])
                data_to_plot2 = self.grid_data[indice]
                data_to_plot2[~self.grid_mask[indice]] = 0
                im = map.pcolormesh(x, y, data_to_plot2, shading="auto")

                if type(self.frame) == np.ndarray:
                    x_boundary, y_boundary = map(np.array([self.frame[0, 0], self.frame[0, 1],
                                                           self.frame[0, 1], self.frame[0, 0],
                                                           self.frame[0, 0]]),
                                                 np.array([self.frame[1, 1], self.frame[1, 1],
                                                           self.frame[1, 0], self.frame[1, 0],
                                                           self.frame[1, 1]]))
                    map.plot(x_boundary, y_boundary, color='m', linewidth=10, marker=None)

                axes.set_title("Data at time " + str(self.t_data[indice])+".")
                plt.colorbar(im, ax=axes)
                plt.show()

            else:
                raise Exception("Can not plot with pcolormesh if the data is not gridded.")

        if type_plot == "contourf":

            if self.grid:
                fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(25, 12))
                if focus:
                    if type(self.frame) == np.ndarray:
                        map = Basemap(projection='cyl', llcrnrlon=self.frame[0, 0], llcrnrlat=self.frame[1, 0],
                                      urcrnrlon=self.frame[0, 1], urcrnrlat=self.frame[1, 1], ax=axes)
                else:
                    map = Basemap(projection='cyl', llcrnrlon=llon, llcrnrlat=llat,
                                  urcrnrlon=ulon, urcrnrlat=ulat, ax=axes)
                map.drawcoastlines()
                x,y = map(self.grid_p_data[0], self.grid_p_data[1])
                data_to_plot2 = self.grid_data[indice]
                data_to_plot2[~self.grid_mask[indice]] = 0
                im = map.contourf(x, y, data_to_plot2)

                if type(self.frame) == np.ndarray:
                    x_boundary, y_boundary = map(np.array([self.frame[0, 0], self.frame[0, 1],
                                                           self.frame[0, 1], self.frame[0, 0],
                                                           self.frame[0, 0]]),
                                                 np.array([self.frame[1, 1], self.frame[1, 1],
                                                           self.frame[1, 0], self.frame[1, 0],
                                                           self.frame[1, 1]]))
                    map.plot(x_boundary, y_boundary, color='m', linewidth=10, marker=None)

                axes.set_title("Data at time " + str(self.t_data[indice])+".")
                plt.colorbar(im, ax=axes)
                plt.show()

            else:
                raise Exception("Can not plot with contourf if the data is not gridded.")

