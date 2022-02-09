from SFD import *
from SFD_graph import *
plt.rcParams.update({'font.size':30})

######creation of the data

def fun(xyz, t):
        return ((xyz[:, 0] * xyz[:, 1] * xyz[:, 2]) * 18) * np.cos(t * 2 * np.pi / 12) +\
                 ((xyz[:, 0] * xyz[:, 1] + xyz[:, 2] * 0.2) * (1 + 0.07 * t))
def funseas(xyz, t):
    return ((xyz[:, 0] * xyz[:, 1] * xyz[:, 2]) * 18) * np.cos(t * 2 * np.pi / 12)

def funtrend(xyz, t):
    return ((xyz[:, 0] * xyz[:, 1] + xyz[:, 2] * 0.2) * (1 + 0.07 * t))

#spatial sampling points
lat = np.arange(-45, 45, 6)
long = np.arange(-90, 90, 6)
latlong = np.meshgrid(lat, long)
longlat = np.array([latlong[1], latlong[0]])
xyz = cart((latlong[1].reshape(-1) + 180) * np.pi / 180, (latlong[0].reshape(-1) + 90) * np.pi / 180)

#time sampling points
t = np.arange(0, 12 * 10, 1)

#data
data = np.zeros((12 * 10, latlong[0].shape[0], latlong[0].shape[1]))
for i in range(12 * 10):
    data[i] = fun(xyz, t[i]).reshape(latlong[0].shape)

#mask
mask = np.ones((12 * 10, latlong[0].shape[0], latlong[0].shape[1]), dtype=bool)

#######setting of the algo

#special format for our data
cdat=cdata(p_data=longlat, t_data=t, data=data, mask=mask, grid=True, frame=np.array([[-92, 87],[-47, 42]]))

cdat.spatial_plot(0, type_plot="pcolormesh", focus=True)

#construction of our field of splines used for the trend-seasonal spatio-temporal approximation
splines = [spline("matern1.5", 0.07), spline("matern1.5", 1.6), spline("matern1.5", 0.07), spline("matern1.5", 9)]
field_psi_seas = spline_field(splines[0], domain="spherical", size=cdat.p_data.shape[0] * 10)
field_phi_seas = spline_field(splines[1], domain="circular", size=12)
field_psi_trend = spline_field(splines[2], domain="spherical", size=cdat.p_data.shape[0] * 10)
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

def fig9_14_1():
    model.data.spatial_plot(0, type_plot="pcolormesh")

def fig9_14_2():
    model.data.spatial_plot(0, type_plot="pcolormesh")

#fit the model to the data with the primal dual splitting method
model.fit(theta=0.5, accuracy_threshold=8e-5, lambda_=lambda_, max_iter=2000, method="PDS",
          sparsity_n=2000, sparsity_levels=True, sparsity_tol=[1e-0, 1e-1, 1e-2, 1e-3, 0])
sparsity_trend_PDS = model.sparsity_trend
sparsity_seas_PDS = model.sparsity_seas

model.fit(theta=0.5, accuracy_threshold=8e-5, lambda_=lambda_, max_iter=2000, method="APGD",
          sparsity_n=2000, sparsity_levels=True, sparsity_tol=[1e-0, 1e-1, 1e-2, 1e-3, 0])
sparsity_trend_APGD = model.sparsity_trend
sparsity_seas_APGD = model.sparsity_seas

def fig9_15_1():
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(25, 12))
    axe.plot(np.arange(0, 2000, 1), sparsity_trend_APGD[:, 0], label="precision = " + str(1e-0) + ".")
    axe.plot(np.arange(0, 2000, 1), sparsity_trend_APGD[:, 1], label="precision = " + str(1e-1) + ".")
    axe.plot(np.arange(0, 2000, 1), sparsity_trend_APGD[:, 2], label="precision = " + str(1e-2) + ".")
    axe.plot(np.arange(0, 2000, 1), sparsity_trend_APGD[:, 3], label="precision = " + str(1e-3) + ".")
    axe.plot(np.arange(0, 2000, 1), sparsity_trend_APGD[:, 4], label="precision = " + str(0) + ".")
    axe.axhline(sparsity_trend_APGD[1999, 2], color="k")
    axe.axhline(sparsity_trend_APGD[1999, 3], color="k")
    plt.yticks([0, sparsity_trend_APGD[1999, 2], 0.05, sparsity_trend_APGD[1999, 3], 0.10, 0.15, 0.2])
    axe.set_xlabel("Iterations")
    axe.set_ylabel("% of entries of weights_trend bigger than precision.")
    plt.legend()

def fig9_15_2():
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(25, 12))
    axe.plot(np.arange(0, 2000, 1), sparsity_seas_APGD[:, 0], label="precision = " + str(1e-0) + ".")
    axe.plot(np.arange(0, 2000, 1), sparsity_seas_APGD[:, 1], label="precision = " + str(1e-1) + ".")
    axe.plot(np.arange(0, 2000, 1), sparsity_seas_APGD[:, 2], label="precision = " + str(1e-2) + ".")
    axe.plot(np.arange(0, 2000, 1), sparsity_seas_APGD[:, 3], label="precision = " + str(1e-3) + ".")
    axe.plot(np.arange(0, 2000, 1), sparsity_seas_APGD[:, 4], label="precision = " + str(0) + ".")
    axe.set_xlabel("Iterations")
    axe.set_ylabel("% of entries of weights_seas bigger than precision.")
    axe.axhline(sparsity_seas_APGD[1999, 1], color="k")
    axe.axhline(sparsity_seas_APGD[1999, 2], color="k")
    plt.yticks([0, sparsity_seas_APGD[1999, 1], 0.05, 0.10, sparsity_seas_APGD[1999, 2], 0.15, 0.2, 0.25])
    plt.legend()
    plt.show()

def fig9_16_1():
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(25, 12))
    axe.plot(np.arange(0, 2000, 1), sparsity_trend_PDS[:, 0], label="precision = " + str(1e-0) + ".")
    axe.plot(np.arange(0, 2000, 1), sparsity_trend_PDS[:, 1], label="precision = " + str(1e-1) + ".")
    axe.plot(np.arange(0, 2000, 1), sparsity_trend_PDS[:, 2], label="precision = " + str(1e-2) + ".")
    axe.plot(np.arange(0, 2000, 1), sparsity_trend_PDS[:, 3], label="precision = " + str(1e-3) + ".")
    axe.plot(np.arange(0, 2000, 1), sparsity_trend_PDS[:, 4], label="precision = " + str(0) + ".")
    axe.axhline(sparsity_trend_PDS[1999, 2], color="k")
    axe.axhline(sparsity_trend_PDS[1999, 3], color="k")
    plt.yticks([0, sparsity_trend_PDS[1999, 2], 0.05, sparsity_trend_PDS[1999, 3], 0.10, 0.15, 0.2, 0.25])
    axe.set_xlabel("Iterations")
    axe.set_ylabel("% of entries of weights_trend bigger than precision.")
    plt.legend()

def fig9_16_2():
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(25, 12))
    axe.plot(np.arange(0, 2000, 1), sparsity_seas_PDS[:, 0], label="precision = " + str(1e-0) + ".")
    axe.plot(np.arange(0, 2000, 1), sparsity_seas_PDS[:, 1], label="precision = " + str(1e-1) + ".")
    axe.plot(np.arange(0, 2000, 1), sparsity_seas_PDS[:, 2], label="precision = " + str(1e-2) + ".")
    axe.plot(np.arange(0, 2000, 1), sparsity_seas_PDS[:, 3], label="precision = " + str(1e-3) + ".")
    axe.plot(np.arange(0, 2000, 1), sparsity_seas_PDS[:, 4], label="precision = " + str(0) + ".")
    axe.set_xlabel("Iterations")
    axe.set_ylabel("% of entries of weights_seas bigger than precision.")
    axe.axhline(sparsity_seas_PDS[1999, 1], color="k")
    axe.axhline(sparsity_seas_PDS[1999, 2], color="k")
    plt.yticks([0, sparsity_seas_PDS[1999, 1], 0.05, 0.10, sparsity_seas_PDS[1999, 2], 0.15, 0.2, 0.25])
    plt.legend()
    plt.show()


