from SFD import *
from SFD_graph import *
plt.rcParams.update({'font.size':25})

######creation of the data

def fun(xyz, t):
        return ((xyz[:, 0] * xyz[:, 1] * xyz[:, 2]) * 18) * np.cos(t * 2 * np.pi / 12) +\
                 ((xyz[:, 0] * xyz[:, 1] + xyz[:, 2] * 0.2) * (1 + 0.07 * t))

def funseas(xyz, t):
    return ((xyz[:, 0] * xyz[:, 1] * xyz[:, 2]) * 18) * np.cos(t * 2 * np.pi / 12)

def funtrend(xyz, t):
    return ((xyz[:, 0] * xyz[:, 1] + xyz[:, 2] * 0.2) * (1 + 0.07 * t))

#spatial sampling points
lat = np.arange(-90, 96, 6)
long = np.arange(-180, 180, 6)
latlong = np.meshgrid(lat, long)
longlat = np.array([latlong[1], latlong[0]])
xyz = cart((latlong[1].reshape(-1) + 180) * np.pi / 180, (latlong[0].reshape(-1) + 90) * np.pi / 180)

#time sampling points
t = np.arange(0, 12 * 10, 1)

#data
data = np.zeros((12 * 10, latlong[0].shape[0], latlong[0].shape[1]))
for i in range(12 * 10):
    data[i] = fun(xyz, t[i]).reshape(latlong[0].shape)

np.random.seed(69)
data_noise = data + np.random.randn(data.shape[0], data.shape[1], data.shape[2]) * data / 10

#mask
mask = np.ones((12 * 10, latlong[0].shape[0], latlong[0].shape[1]), dtype=bool)

#######setting of the algo

#special format for our data
cdat=cdata(p_data=longlat, t_data=t, data=data_noise, mask=mask, grid=True)

#construction of our field of splines used for the trend-seasonal spatio-temporal approximation
splines = [spline("matern1.5", 0.14), spline("matern1.5", 1.6), spline("matern1.5", 0.14), spline("matern1.5", 9)]
field_psi_seas = spline_field(splines[0], domain="spherical", size=cdat.p_data.shape[0])
field_phi_seas = spline_field(splines[1], domain="circular", size=12)
field_psi_trend = spline_field(splines[2], domain="spherical", size=cdat.p_data.shape[0])
field_zeta_trend = spline_field(splines[3], size=6 * 10,
        domain=np.array([cdat.t_data[0] - 0.2 * cdat.M * min_increment(cdat.t_data),
                         cdat.t_data[-1] + 0.2 * cdat.M * min_increment(cdat.t_data)]))

#instance of the class sfd, creating a model for fitting the data
model = sfd(data=cdat, splines=[field_psi_seas, field_phi_seas, field_psi_trend, field_zeta_trend], T=12)

#build different matrices used for the fit (discretization ones)
model.build(code=["seas", "trend"])

#get a good lambda (regularization)
x_eval = np.zeros(model.dim)
l22_loss = SquaredL2Loss(dim=model.data.true_dim, data=model.data.data[model.data.mask]) * model.F
lambda_ = 0.001 * np.max(np.abs(l22_loss.gradient(0 * x_eval)))

#fit the model to the data with the primal dual splitting method
model.fit(theta=0.5, accuracy_threshold=8e-5, lambda_=lambda_, max_iter=10, method="PDS",
          sparsity_n=10, sparsity_levels=True, sparsity_tol=[1e-1, 1e-2, 1e-3, 0])

#graphical model for interpretation
model_graph = sfd_graph(model)
model_graph.temporal_plot(indice=400)

def fig9_10():

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 12))
    map0 = Basemap(projection='cyl', llcrnrlon=-180, llcrnrlat=-70,
                  urcrnrlon=180, urcrnrlat=70, ax=axes[0])
    map1 = Basemap(projection='cyl', llcrnrlon=-180, llcrnrlat=-70,
                  urcrnrlon=180, urcrnrlat=70, ax=axes[1])
    map0.drawcoastlines()
    map1.drawcoastlines()
    x0, y0 = map0(longlat[0], longlat[1])
    x1, y1 = map1(longlat[0], longlat[1])
    im0 = map0.pcolormesh(x0, y0, data[12], shading="auto")
    im1 = map1.pcolormesh(x1, y1, data_noise[12], shading="auto")
    plt.colorbar(im0, ax=axes[0], shrink=0.3)
    plt.colorbar(im1, ax=axes[1], shrink=0.3)
    axes[0].set_title("Plot of the spatial data field at time 12.")
    axes[1].set_title("Plot of the spatial noisy data field at time 12.")
    plt.show()

def fig9_11():
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(25, 12))
    axes.plot(t, data[:, 28, 14], label="Linear interpolation of the data.")
    axes.plot(t, data_noise[:, 28, 14], label="Linear interpolation of the noisy data.")
    axes.scatter(t, data[:, 28, 14], label="Data.")
    axes.scatter(t, data_noise[:, 28, 14], label="Noisy data.")
    plt.xlabel("Time")
    plt.ylabel("Value of f([0.45278918, 0.2879105 , 0.84385396],Time)")
    plt.legend()
    plt.show()

def fig9_12():

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 12))
    map0 = Basemap(projection='cyl', llcrnrlon=-180, llcrnrlat=-70,
                  urcrnrlon=180, urcrnrlat=70, ax=axes[0])
    map1 = Basemap(projection='cyl', llcrnrlon=-180, llcrnrlat=-70,
                  urcrnrlon=180, urcrnrlat=70, ax=axes[1])
    map0.drawcoastlines()
    map1.drawcoastlines()
    x0, y0 = map0(longlat[0], longlat[1])
    x1, y1 = map1(longlat[0], longlat[1])
    im0 = map0.pcolormesh(x0, y0, data[12], shading="auto")
    im1 = map1.pcolormesh(x1, y1, model.evaluation(xyz=xyz, t=12, code=["trend", "seas"]).reshape(60, 31), shading="auto")
    plt.colorbar(im0, ax=axes[0], shrink=0.3)
    plt.colorbar(im1, ax=axes[1], shrink=0.3)
    axes[0].set_title("Spatial un-noised data (original) at time 12.")
    axes[1].set_title("Trend+seas fit of the data at time 12.")
    plt.show()

    model_graph.spatial_plot(t=12, type_plot="pcolormesh", code=["data", "residuals"])

def fig9_13():
     model_graph.temporal_plot(indice=400, code=["seas", "trend", "dada", "trendseas"])

def get_metrics():

    t = np.array(np.arange(0, 120, 1), dtype=float)
    errors_seas = np.zeros((t.shape[0], longlat[0].shape[0] * longlat[0].shape[1]))
    errors_trend = np.zeros((t.shape[0], longlat[0].shape[0] * longlat[0].shape[1]))
    bfun = funtrend(xyz, 0)
    bev = model.evaluation(xyz=xyz, t=0, code=["trend"]).reshape(-1)

    for i in range(t.shape[0]):
        errors_seas[i] = funseas(xyz, t[i]) - model.evaluation(xyz=xyz, t=t[i], code=["seas"]).reshape(-1) - \
                         bev + bfun
        errors_trend[i] = funtrend(xyz, t[i]) - model.evaluation(xyz=xyz, t=t[i], code=["trend"]).reshape(-1) + \
                         bev - bfun

    metrics = np.array([[np.max(np.abs(errors_seas)), np.mean(np.abs(errors_seas)), np.var(np.abs(errors_seas))],
                       [np.max(np.abs(errors_trend)), np.mean(np.abs(errors_trend)), np.var(np.abs(errors_trend))]])

    return metrics



