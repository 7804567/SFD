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

#mask
mask = np.ones((12 * 10, latlong[0].shape[0], latlong[0].shape[1]), dtype=bool)

#introducing forced sparsity in the data
np.random.seed(69)
mask = np.hstack((np.zeros(int(latlong[0].shape[0] * latlong[0].shape[1] * 10 * 12 / 10), dtype=int),
                                       np.ones(int(latlong[0].shape[1] * latlong[0].shape[0] * 9 * 10 * 12 / 10), dtype=int)))
np.random.shuffle(mask)
mask = mask.reshape(10 * 12, latlong[0].shape[0], latlong[0].shape[1])

#######setting of the algo

#special format for our data
cdat=cdata(p_data=longlat, t_data=t, data=data, mask=mask, grid=True)

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
model.fit(theta=0.5, accuracy_threshold=8e-5, lambda_=lambda_, max_iter=800, method="PDS")


#vizualisation
model_graph = sfd_graph(model)

def fig9_7_1():
    cdat.spatial_plot(indice=0, type_plot="pcolormesh")

def fig9_7_2():
    cdat.spatial_plot(indice=66, type_plot="pcolormesh")

def fig9_8_1():
    model_graph.spatial_plot(12, code=["trendseas", "data"])

def fig9_8_2():
    model_graph.spatial_plot(12, code=["residuals"])

def fig9_9():
    model_graph.temporal_plot(400, code=["data", "residuals", "trendseas"])


def model_efficiency1():

    t = np.arange(0, 12 * 10, 1)
    mask = np.ones((12 * 10, latlong[0].shape[0], latlong[0].shape[1]), dtype=bool)
    np.random.seed(69)
    mask = np.hstack((np.zeros(int(latlong[0].shape[0] * latlong[0].shape[1] * 10 * 12 / 10), dtype=int),
                      np.ones(int(latlong[0].shape[1] * latlong[0].shape[0] * 9 * 10 * 12 / 10), dtype=int)))
    np.random.shuffle(mask)
    mask = mask.reshape(10 * 12, latlong[0].shape[0], latlong[0].shape[1])
    mask = np.array(mask, dtype=bool)
    errors_trend_data = np.zeros(sum(sum(sum(mask))))
    errors_seas_data = np.zeros(sum(sum(sum(mask))))
    CI_trend_data, CI_seas_data = 0, 0
    xyz = cart((longlat[0].reshape(-1) + 180) * np.pi / 180, (longlat[1].reshape(-1) + 90) * np.pi / 180)
    t = np.array(t, dtype=float)
    bfun = funtrend(xyz, 0)
    bev = model.evaluation(xyz=xyz, t=0, code=["trend"]).reshape(-1)

    for i in range(t.shape[0]):
        print("STEP " + str(i) + "done")
        indice=mask[i].reshape(-1)

        errors_trend_data[CI_trend_data: CI_trend_data + sum(indice)] = \
            funtrend(xyz[indice], t[i]) - model.evaluation(xyz=xyz[indice], t=t[i], code=["trend"]).reshape(-1) + \
        bev[indice] - bfun[indice]
        CI_trend_data += sum(indice)

        errors_seas_data[CI_seas_data: CI_seas_data + sum(indice)] = \
            funseas(xyz[indice], t[i]) - model.evaluation(xyz=xyz[indice], t=t[i], code=["seas"]).reshape(-1) - \
            bev[indice] + bfun[indice]
        CI_seas_data += sum(indice)

    return np.max(np.abs(errors_seas_data)), np.mean(np.abs(errors_seas_data)), np.var(np.abs(errors_seas_data)), \
           np.max(np.abs(errors_trend_data)), np.mean(np.abs(errors_trend_data)),np.var(np.abs(errors_trend_data))

def model_efficiency2():

    t = np.arange(0, 12 * 10, 1)
    mask = np.ones((12 * 10, latlong[0].shape[0], latlong[0].shape[1]), dtype=bool)
    np.random.seed(69)
    mask = np.hstack((np.zeros(int(latlong[0].shape[0] * latlong[0].shape[1] * 10 * 12 / 10), dtype=int),
                        np.ones(int(latlong[0].shape[1] * latlong[0].shape[0] * 9 * 10 * 12 / 10), dtype=int)))
    np.random.shuffle(mask)
    mask = mask.reshape(10 * 12, latlong[0].shape[0], latlong[0].shape[1])
    mask = np.array(mask, dtype=bool)
    errors_trend_datac = np.zeros(120 * lat.shape[0] * long.shape[0] - sum(sum(sum(mask))))
    errors_seas_datac = np.zeros(120 * lat.shape[0] * long.shape[0] - sum(sum(sum(mask))))
    CI_trend_datac, CI_seas_datac = 0, 0
    xyz = cart((longlat[0].reshape(-1) + 180) * np.pi / 180, (longlat[1].reshape(-1) + 90) * np.pi / 180)
    t = np.array(t, dtype=float)
    bfun = funtrend(xyz, 0)
    bev = model.evaluation(xyz=xyz, t=0, code=["trend"]).reshape(-1)

    for i in range(t.shape[0]):
        print("STEP " + str(i) + "done")
        indice = mask[i].reshape(-1)

        errors_trend_datac[CI_trend_datac: CI_trend_datac + sum(~indice)] = \
            funtrend(xyz[~indice], t[i]) - model.evaluation(xyz=xyz[~indice], t=t[i], code=["trend"]).reshape(-1) + \
            bev[~indice] - bfun[~indice]
        CI_trend_datac += sum(~indice)

        errors_seas_datac[CI_seas_datac: CI_seas_datac + sum(~indice)] = \
            funseas(xyz[~indice], t[i]) - model.evaluation(xyz=xyz[~indice], t=t[i], code=["seas"]).reshape(-1) - \
            bev[~indice] + bfun[~indice]
        CI_seas_datac += sum(~indice)

    return np.max(np.abs(errors_seas_datac)), np.mean(np.abs(errors_seas_datac)), np.var(np.abs(errors_seas_datac)), \
           np.max(np.abs(errors_trend_datac)), np.mean(np.abs(errors_trend_datac)), np.var(np.abs(errors_trend_datac))

def model_efficiency3():

    p = sphericallattice_Fibonacci(5000)
    t2 = np.linspace(0, 120, 500)
    errors_trend_unif = np.zeros((t2.shape[0], p.shape[0]))
    errors_seas_unif = np.zeros((t2.shape[0], p.shape[0]))
    bfun = funtrend(p, 0)
    bev = model.evaluation(xyz=p, t=0, code=["trend"]).reshape(-1)
    for i in range(t2.shape[0]):
        print("step " + str(i) +" done.")
        errors_seas_unif[i] = funseas(p, t2[i]) - model.evaluation(xyz=p, t=t2[i], code=["seas"]).reshape(-1) + \
            bfun - bev
        errors_trend_unif[i] = funtrend(p, t2[i]) - model.evaluation(xyz=p, t=t2[i], code=["trend"]).reshape(-1) + \
            bev - bfun

    return np.max(np.abs(errors_seas_unif[:-5])), np.mean(np.abs(errors_seas_unif[:-5])), \
           np.var(np.abs(errors_seas_unif[:-5])), np.max(np.abs(errors_trend_unif[:-5])), \
           np.mean(np.abs(errors_trend_unif[:-5])), np.var(np.abs(errors_trend_unif[:-5]))
