import numpy as np
import scipy.sparse as sp
from pycsou.linop.base import SparseLinearOperator, LinOpStack
from pycsou.func.penalty import L1Norm
from pycsou.func.loss import SquaredL2Loss
from pycsou.linop.base import KroneckerProduct
from pycsou.core.functional import ProximableFunctional
from pycsou.opt.proxalgs import APGD, PDS
import matplotlib.pyplot as plt
import os


from splines_code import *

from utility import *

from utility_fun import *

from cdata_code import *

class L1K (ProximableFunctional):

    def __init__(self, K, dim):
        self.K = np.abs(K)
        super().__init__(dim)

    def __call__(self, x):
        return self.K.dot(np.abs(x))

    def prox(self, x, tau):
        return np.sign(x) * np.maximum(np.abs(x) - self.K * tau, 0)


class sfd:

    def __init__(self, data=None, splines=None, size="none", T=1, sparse=True, padd=0.2, folder_name=None):

        if folder_name == None:
            self.data = data
            self.T = T
            self.padd = padd
            self.fitted = False

            #splines should be a list of splines of the form:
                #(psi_seas, phi_seas, psi_trend, rho_trend)
            #psi corresponds to a spherical spline
            #phi corresponds to a circular spline
            #rho correpinds to a real spline

            #splines can also be a list of field of splines

            if isinstance(splines[0], spline) and not isinstance(splines[0], spline_field):
                self.N_seas = size[0]
                self.M_seas = size[1]
                self.N_trend = size[2]
                self.M_trend = size[3]

                self.field_psi_seas = spline_field(spline=splines[0], size=self.N_seas, domain="spherical")
                self.field_phi_seas = spline_field(spline=splines[1], size=self.M_seas, domain="circular")
                self.field_psi_trend = spline_field(spline=splines[2], size=self.N_trend, domain="spherical")
                self.field_zeta_trend = spline_field(spline=splines[3], size=self.M_trend,
                                domain=np.array([self.data.t_data[0] - self.padd * self.data.M * min_increment(self.data.t_data),
                                                self.data.t_data[-1] + self.padd * self.data.M * min_increment(self.data.t_data)]))

            if isinstance(splines[0], spline_field):

                self.field_psi_seas = splines[0]
                self.field_phi_seas = splines[1]
                self.field_psi_trend = splines[2]
                self.field_zeta_trend = splines[3]

                self.N_seas = self.field_psi_seas.size
                self.M_seas = self.field_phi_seas.size
                self.N_trend = self.field_psi_trend.size
                self.M_trend = self.field_zeta_trend.size

            #p corresponds to points on the sphere
            #c corresponds to points on the circle
            #t correponds to points in time
            #if p or c or t is followed by _seas or _trend
                # it means that they are nodes (fixed) used for the splines

        else:
            if isinstance(folder_name, str):
                path = os.path.join(os.getcwd(), folder_name)
                if os.path.exists(path):
                    self.load(folder_name=folder_name)

    def build(self, code=["trend"], tol=1e-3):

        self.code = code
        tol = tol

        if "seas" in self.code:

            self.PSI_seas = self.field_psi_seas.evaluate(self.field_psi_seas.proj(np.clip(
                self.data.p_data.dot(self.field_psi_seas.nodes.transpose()), -1, 1)))
            self.PSI_seas[(self.PSI_seas < tol)] = 0
            self.PSI_seas = SparseLinearOperator(sp.csr_matrix(self.PSI_seas))

            self.PHI_seas = self.field_phi_seas.evaluate(self.field_phi_seas.proj(np.clip(
                proj_circle(self.T, self.data.t_data).dot(self.field_phi_seas.nodes.transpose()), -1, 1)))
            self.PHI_seas[(self.PHI_seas < tol)] = 0
            self.PHI_seas = SparseLinearOperator(sp.csr_matrix(self.PHI_seas))

            self.op_seas = KroneckerProduct(self.PSI_seas, self.PHI_seas)

        if "trend" in self.code:

            self.PSI_trend = self.field_psi_trend.evaluate(self.field_psi_trend.proj(np.clip(
                self.data.p_data.dot(self.field_psi_trend.nodes.transpose()), -1, 1)))
            self.PSI_trend[(self.PSI_trend < tol)] = 0
            self.PSI_trend = SparseLinearOperator(sp.csr_matrix(self.PSI_trend))

            self.ZETA_trend = self.field_zeta_trend.evaluate(self.field_zeta_trend.proj(self.data.t_data.reshape(self.data.M, 1) -
                                                     self.field_zeta_trend.nodes.reshape(1, self.M_trend)))
            self.ZETA_trend[(self.ZETA_trend < tol)] = 0
            self.ZETA_trend = SparseLinearOperator(sp.csr_matrix(self.ZETA_trend))

            self.op_trend = KroneckerProduct(self.PSI_trend, self.ZETA_trend)


        if "trend" in self.code:
            if "seas" in self.code:
                self.dim = self.N_seas * self.M_seas + self.N_trend * self.M_trend
                self.F = LinOpStack(SparseLinearOperator(self.data.Mask) * self.op_seas,
                                    SparseLinearOperator(self.data.Mask) * self.op_trend, axis=1)
            else:
                self.dim = self.N_trend * self.M_trend
                self.F = SparseLinearOperator(self.data.Mask) * self.op_trend
        else:
            self.dim = self.N_seas * self.M_seas
            self.F = SparseLinearOperator(self.data.Mask) * self.op_seas

        self.F.compute_lipschitz_cst(tol=1e-3)

    def fit(self, lambda_="auto", theta=0.5, max_iter=500, accuracy_threshold=1e-3, method="APGD",
            sparsity_levels=False, sparsity_tol=0, sparsity_n=0):

        self.accuracy_threshold = accuracy_threshold
        self.max_iter = max_iter

        if lambda_ == "auto":
            x_eval = np.zeros(self.dim)
            l22_loss = SquaredL2Loss(dim=self.data.true_dim, data=self.data.data[self.data.mask]) * self.F
            self.lambda_ = 0.01 * np.max(np.abs(l22_loss.gradient(0 * x_eval)))
        else:
            self.lambda_ = lambda_
        self.theta = theta


        if "trend" in self.code:
            if "seas" in self.code:
                d = np.hstack((np.ones(self.N_seas * self.M_seas) * self.theta * self.lambda_,
                               np.ones(self.N_trend * self.M_trend) * (1 - self.theta) * self.lambda_))
            else:
                d = np.ones(self.M_trend * self.N_trend) * self.lambda_
        else:
            d = np.ones(self.M_seas * self.N_seas) * self.lambda_

        if method == "APGD":
            self.fitting_method = "APGD"
            self.G = L1K(d, self.dim)

            l22_loss = SquaredL2Loss(dim=self.data.true_dim, data=self.data.data[self.data.mask]) * self.F
            iterative_method = APGD(dim=self.dim, F=l22_loss, G=self.G, verbose=1, acceleration="CD",
                                                     max_iter=max_iter, accuracy_threshold=accuracy_threshold)
            key = "iterand"


        if method == "PDS":
            self.fitting_method = "PDS"
            self.G = None
            self.H = L1Norm(dim=self.dim)
            self.K = SparseLinearOperator(sp.diags(d, 0))
            self.K.compute_lipschitz_cst()

            l22_loss = SquaredL2Loss(dim=self.data.true_dim, data=self.data.data[self.data.mask]) * self.F
            iterative_method = PDS(dim=self.dim, F=l22_loss, G=self.G, H=self.H, K=self.K, verbose=1, max_iter=max_iter,
                        accuracy_threshold=accuracy_threshold)
            key = "primal_variable"

        if not sparsity_levels:

            estimate, converged, diagnostics = iterative_method.iterate()
            self.converged = converged
            self.diagnostics = diagnostics

            if "trend" in self.code:
                if "seas" in self.code:
                    self.estimate_seas = estimate[key][0:(self.N_seas * self.M_seas)] \
                        .reshape(self.M_seas, self.N_seas).transpose()
                    self.estimate_trend = estimate[key][(self.N_seas * self.M_seas):] \
                        .reshape(self.M_trend, self.N_trend).transpose()
                else:
                    self.estimate_trend = estimate[key].reshape(self.M_trend, self.N_trend).transpose()
            else:
                self.estimate_seas = estimate[key].reshape(self.M_seas, self.N_seas).transpose()

        if sparsity_levels:

            sparsity_tol = np.array([sparsity_tol]).reshape(-1)
            n=0

            if "trend" in self.code:
                self.sparsity_trend = np.zeros((sparsity_n, sparsity_tol.shape[0]))
                dim_trend = float(self.M_trend * self.N_trend)

            if "seas" in self.code:
                self.sparsity_seas = np.zeros((sparsity_n, sparsity_tol.shape[0]))
                dim_seas = float(self.M_seas * self.N_seas)

            for iterand_o in iterative_method.iterates(n=sparsity_n):

                iterand = np.copy(iterand_o)

                if "trend" in self.code:
                    if "seas" in self.code:
                        estimate_seas = iterand.tolist()[key][0:self.N_seas * self.M_seas]
                        estimate_trend = iterand.tolist()[key][self.N_seas * self.M_seas:]
                    else:
                        estimate_trend = iterand.tolist()[key]
                else:
                    estimate_seas = iterand.tolist()[key]

                if "trend" in self.code:
                    for i in range(sparsity_tol.shape[0]):
                        self.sparsity_trend[n, i] = np.sum((estimate_trend > sparsity_tol[i])) / dim_trend

                if "seas" in self.code:
                    for i in range(sparsity_tol.shape[0]):
                        self.sparsity_seas[n, i] = np.sum((estimate_seas > sparsity_tol[i])) / dim_seas

                print("Iter " + str(n+1) + "/" + str(sparsity_n) + ".")
                n+=1

                if n == sparsity_n:

                    if "trend" in self.code:
                        if "seas" in self.code:
                            self.estimate_seas = iterand.tolist()[key][0:self.N_seas * self.M_seas].reshape(self.M_seas,
                                                                                            self.N_seas).transpose()
                            self.estimate_trend = iterand.tolist()[key][self.N_seas * self.M_seas:].reshape(self.M_trend,
                                                                                            self.N_trend).transpose()
                        else:
                            self.estimate_trend = iterand.tolist()[key].reshape(self.M_trend, self.N_trend).transpose()
                    else:
                        self.estimate_seas = iterand.tolist()[key].reshape(self.M_seas, self.N_seas).transpose()

        self.fitted = True

    def evaluation(self, xyz, t, weights_seas="none", weights_trend="none", code=["seas", "trend"]):

        if isinstance(xyz, np.ndarray) and xyz.shape == (3,):
            eval = np.zeros(t.shape)
            evaluation_type = "temporal"

        elif isinstance(t, float) or isinstance(t, int) or (isinstance(t, np.ndarray) and t.shape == (1, )):
            eval = np.zeros((xyz.shape[0], 1))
            evaluation_type = "spatial"

        else:
            eval = np.zeros((xyz.shape[0], t.shape[0]))
         #raise Exception("Your evaluation format is not supported: xyz or t must be 0 dimensional (a point).")

        if "seas" in code:

            if weights_seas == "none":
                eval += self.field_psi_seas.evaluate(self.field_psi_seas.proj(np.clip(
                    xyz.dot(self.field_psi_seas.nodes.transpose()), -1, 1))) \
                    .dot(self.estimate_seas).dot(self.field_phi_seas.evaluate(self.field_phi_seas.proj(np.clip(
                    self.field_phi_seas.nodes.dot(proj_circle(self.T, t).transpose()), -1, 1))))
            else:
                eval += self.field_psi_seas.evaluate(self.field_psi_seas.proj(np.clip(
                    xyz.dot(self.field_psi_seas.nodes.transpose()), -1, 1))) \
                    .dot(weights_seas).dot(self.field_phi_seas.evaluate(self.field_phi_seas.proj(np.clip(
                    self.field_phi_seas.nodes.dot(proj_circle(self.T, t).transpose()), -1, 1))))

        if "trend" in code:

            if testreal(t):
                t = np.array([t])

            if weights_trend == "none":
                eval += self.field_psi_trend.evaluate(self.field_psi_trend.proj(np.clip(
                    xyz.dot(self.field_psi_trend.nodes.transpose()), -1, 1))) \
                    .dot(self.estimate_trend).dot(self.field_zeta_trend.evaluate(self.field_zeta_trend.proj(
                    self.field_zeta_trend.nodes.reshape(self.M_trend, 1) - t.reshape(1, t.shape[0]))))
            else:
                eval += self.field_psi_trend.evaluate(self.field_psi_trend.proj(np.clip(
                    xyz.dot(self.field_psi_trend.nodes.transpose()), -1, 1))) \
                    .dot(weights_trend).dot(self.field_zeta_trend.evaluate(self.field_zeta_trend.proj(
                        self.field_zeta_trend.nodes.reshape(self.M_trend, 1) - t.reshape(1, t.shape[0]))))

        return eval

    def load(self, folder_name):

        path = os.path.join(os.getcwd(), folder_name)

        if os.path.exists(path):

            var = np.load(os.path.join(path, "info_vector.npy"), allow_pickle=True)
            self.N_seas = int(var[0])
            self.M_seas = int(var[1])
            self.N_trend = int(var[2])
            self.M_trend = int(var[3])
            self.T = var[4]
            self.padd = var[5]

            path8 = os.path.join(path, "data")
            grid = np.load(os.path.join(path8, "grid.npy"), allow_pickle=True)[0]
            if not grid:
                self.data = cdata(p_data=np.load(os.path.join(path8, "p_data_SC.npy"), allow_pickle=True),
                                  t_data=np.load(os.path.join(path8, "t_data.npy"), allow_pickle=True),
                                  data=np.load(os.path.join(path8, "data.npy"), allow_pickle=True),
                                  mask=np.load(os.path.join(path8, "mask.npy"), allow_pickle=True),
                                  grid=grid,
                                  frame=np.load(os.path.join(path8, "frame.npy"), allow_pickle=True)[0])
            if grid:
                self.data = cdata(p_data=np.load(os.path.join(path8, "grid_p_data.npy"), allow_pickle=True),
                                  t_data=np.load(os.path.join(path8, "t_data.npy"), allow_pickle=True),
                                  data=np.load(os.path.join(path8, "grid_data.npy"), allow_pickle=True),
                                  mask=np.load(os.path.join(path8, "grid_mask.npy"), allow_pickle=True),
                                  grid=grid,
                                  frame=np.load(os.path.join(path8, "frame.npy"), allow_pickle=True)[0])

            path2 = os.path.join(path, "field_psi_seas")
            var = np.load(os.path.join(path2, "float.npy"), allow_pickle=True)
            var2 = np.load(os.path.join(path2, "specification.npy"), allow_pickle=True)
            var3 = np.load(os.path.join(path2, "domain.npy"), allow_pickle=True)
            var4 = np.load(os.path.join(path2, "frame.npy"), allow_pickle=True)
            if isinstance(var3[0], str):
                var3 = var3[0]
            else:
                var3 = var3.reshape(-1)
            self.field_psi_seas = spline_field(spline=spline(specification=var2[0], eps=var[0]), domain = var3,
                                    size=var[1], nodes=np.load(os.path.join(path2, "nodes_SC.npy"), allow_pickle=True),
                                               frame=var4[0])

            path3 = os.path.join(path, "field_psi_trend")
            var = np.load(os.path.join(path3, "float.npy"), allow_pickle=True)
            var2 = np.load(os.path.join(path3, "specification.npy"), allow_pickle=True)
            var3 = np.load(os.path.join(path3, "domain.npy"), allow_pickle=True)
            var4 = np.load(os.path.join(path3, "frame.npy"), allow_pickle=True)
            if isinstance(var3[0], str):
                var3 = var3[0]
            else:
                var3 = var3.reshape(-1)
            self.field_psi_trend = spline_field(spline=spline(specification=var2[0], eps=var[0]), domain = var3,
                                    size=var[1], nodes=np.load(os.path.join(path3, "nodes_SC.npy"), allow_pickle=True),
                                                frame=var4[0])

            path4 = os.path.join(path, "field_zeta_trend")
            var = np.load(os.path.join(path4, "float.npy"), allow_pickle=True)
            var2 = np.load(os.path.join(path4, "specification.npy"), allow_pickle=True)
            var3 = np.load(os.path.join(path4, "domain.npy"), allow_pickle=True)
            if isinstance(var3[0], str):
                var3 = var3[0]
            else:
                var3 = var3.reshape(-1)
            self.field_zeta_trend = spline_field(spline=spline(specification=var2[0], eps=var[0]), domain = var3,
                                    size=var[1], nodes=np.load(os.path.join(path4, "nodes.npy"), allow_pickle=True))

            path5 = os.path.join(path, "field_phi_seas")
            var = np.load(os.path.join(path5, "float.npy"), allow_pickle=True)
            var2 = np.load(os.path.join(path5, "specification.npy"), allow_pickle=True)
            var3 = np.load(os.path.join(path5, "domain.npy"), allow_pickle=True)
            if isinstance(var3[0], str):
                var3 = var3[0]
            else:
                var3 = var3.reshape(-1)
            self.field_phi_seas = spline_field(spline=spline(specification=var2[0], eps=var[0]), domain = var3,
                                    size=var[1], nodes=np.load(os.path.join(path5, "nodes.npy"), allow_pickle=True))

            if os.path.exists(os.path.join(path, "model_code.npy")):
                self.code = np.load(os.path.join(path, "model_code.npy"), allow_pickle=True).reshape(-1)
                tol = 1e-3

                if "seas" in self.code:
                    self.PSI_seas = self.field_psi_seas.evaluate(self.field_psi_seas.proj(np.clip(
                        self.data.p_data.dot(self.field_psi_seas.nodes.transpose()), -1, 1)))
                    self.PSI_seas[(self.PSI_seas < tol)] = 0
                    self.PSI_seas = SparseLinearOperator(sp.csr_matrix(self.PSI_seas))

                    self.PHI_seas = self.field_phi_seas.evaluate(self.field_phi_seas.proj(np.clip(
                        proj_circle(self.T, self.data.t_data).dot(self.field_phi_seas.nodes.transpose()), -1, 1)))
                    self.PHI_seas[(self.PHI_seas < tol)] = 0
                    self.PHI_seas = SparseLinearOperator(sp.csr_matrix(self.PHI_seas))

                    self.op_seas = KroneckerProduct(self.PSI_seas, self.PHI_seas)

                if "trend" in self.code:
                    self.PSI_trend = self.field_psi_trend.evaluate(self.field_psi_trend.proj(np.clip(
                        self.data.p_data.dot(self.field_psi_trend.nodes.transpose()), -1, 1)))
                    self.PSI_trend[(self.PSI_trend < tol)] = 0
                    self.PSI_trend = SparseLinearOperator(sp.csr_matrix(self.PSI_trend))

                    self.ZETA_trend = self.field_zeta_trend.evaluate(
                        self.field_zeta_trend.proj(self.data.t_data.reshape(self.data.M, 1) -
                                                   self.field_zeta_trend.nodes.reshape(1, self.M_trend)))
                    self.ZETA_trend[(self.ZETA_trend < tol)] = 0
                    self.ZETA_trend = SparseLinearOperator(sp.csr_matrix(self.ZETA_trend))
                    self.op_trend = KroneckerProduct(self.PSI_trend, self.ZETA_trend)

                var = np.load(os.path.join(path, "model_info_vector.npy"), allow_pickle=True)
                self.dim = int(var[1])
                if "trend" in self.code:
                    if "seas" in self.code:
                        self.F = LinOpStack(SparseLinearOperator(self.data.Mask) * self.op_seas,
                                            SparseLinearOperator(self.data.Mask) * self.op_trend, axis=1)
                    else:
                        self.F = SparseLinearOperator(self.data.Mask) * self.op_trend
                else:
                    self.F = SparseLinearOperator(self.data.Mask) * self.op_seas

                self.F.lipschitz_cst = var[0]

            path7 = os.path.join(path, "fit")
            if os.path.exists(path7):
                self.fitted = True
                self.fitting_method = np.load(os.path.join(path7, "fitting_method.npy"), allow_pickle=True)[0]
                var = np.load(os.path.join(path7, "info_vector.npy"), allow_pickle=True)
                self.lambda_ = var[0]
                self.theta = var[1]
                self.accuracy_threshold = var[2]
                self.max_iter = var[3]
                if "seas" in self.code:
                    self.estimate_seas = np.load(os.path.join(path7, "estimate_seas.npy"), allow_pickle=True)
                if "trend" in self.code:
                    self.estimate_trend = np.load(os.path.join(path7, "estimate_trend.npy"), allow_pickle=True)

    def save(self, folder_name):

        if not isinstance(folder_name, str):
            raise Exception("folder_name has to be of type str.")
        else:
            path = os.path.join(os.getcwd(), folder_name)

        os.makedirs(path)

        np.save(os.path.join(path, "info_vector"), np.array([self.N_seas, self.M_seas, self.N_trend, self.M_trend, self.T, self.padd]))

        path1 = os.path.join(path, "field_psi_seas")
        os.makedirs(path1)
        np.save(os.path.join(path1, "float"), np.array([self.field_psi_seas.eps, self.field_psi_seas.size]))
        np.save(os.path.join(path1, "nodes"), self.field_psi_seas.nodes)
        np.save(os.path.join(path1, "nodes_SC"), self.field_psi_seas.nodes_SC)
        np.save(os.path.join(path1, "specification"), np.array([self.field_psi_seas.specification]))
        np.save(os.path.join(path1, "domain"), np.array([self.field_psi_seas.domain]))
        np.save(os.path.join(path1, "frame"), np.array([self.field_psi_seas.frame]))

        path2 = os.path.join(path, "field_psi_trend")
        os.makedirs(path2)
        np.save(os.path.join(path2, "float"), np.array([self.field_psi_trend.eps, self.field_psi_trend.size]))
        np.save(os.path.join(path2, "nodes"), self.field_psi_trend.nodes)
        np.save(os.path.join(path2, "nodes_SC"), self.field_psi_trend.nodes_SC)
        np.save(os.path.join(path2, "specification"), np.array([self.field_psi_trend.specification]))
        np.save(os.path.join(path2, "domain"), np.array([self.field_psi_trend.domain]))
        np.save(os.path.join(path2, "frame"), np.array([self.field_psi_trend.frame]))

        path3 = os.path.join(path, "field_zeta_trend")
        os.makedirs(path3)
        np.save(os.path.join(path3, "float"), np.array([self.field_zeta_trend.eps, self.field_zeta_trend.size]))
        np.save(os.path.join(path3, "nodes"), self.field_zeta_trend.nodes)
        np.save(os.path.join(path3, "specification"), np.array([self.field_zeta_trend.specification]))
        np.save(os.path.join(path3, "domain"), np.array([self.field_zeta_trend.domain]))


        path4 = os.path.join(path, "field_phi_seas")
        os.makedirs(path4)
        np.save(os.path.join(path4, "float"), np.array([self.field_phi_seas.eps, self.field_phi_seas.size]))
        np.save(os.path.join(path4, "nodes"), self.field_phi_seas.nodes)
        np.save(os.path.join(path4, "specification"), np.array([self.field_phi_seas.specification]))
        np.save(os.path.join(path4, "domain"), np.array([self.field_phi_seas.domain]))

        if hasattr(self, "code"):

            np.save(os.path.join(path, "model_code"), np.array([self.code]))
            np.save(os.path.join(path, "model_info_vector"), np.array([self.F.lipschitz_cst, self.dim]))

        if self.fitted:

            path6 = os.path.join(path, "fit")
            os.makedirs(path6)
            np.save(os.path.join(path6, "info_vector"), np.array([self.lambda_, self.theta, self.accuracy_threshold,
                                                                  self.max_iter]))
            np.save(os.path.join(path6, "fitting_method"), np.array([self.fitting_method]))

            if "trend" in self.code:
                np.save(os.path.join(path6, "estimate_trend"), self.estimate_trend)
            if "seas" in self.code:
                np.save(os.path.join(path6, "estimate_seas"), self.estimate_seas)

        path7 = os.path.join(path, "data")
        os.makedirs(path7)
        np.save(os.path.join(path7, "frame"), np.array([self.data.frame]))

        if not self.data.grid:
            np.save(os.path.join(path7, "t_data"), self.data.t_data)
            np.save(os.path.join(path7, "p_data"), self.data.p_data)
            np.save(os.path.join(path7, "p_data_SC"), self.data.p_data_SC)
            np.save(os.path.join(path7, "data"), self.data.data.reshape(self.data.M, self.data.N).transpose())
            np.save(os.path.join(path7, "mask"), self.data.mask.reshape(self.data.M, self.data.N).transpose())
            np.save(os.path.join(path7, "grid"), np.array([self.data.grid]))

        if self.data.grid:
            np.save(os.path.join(path7, "grid_data"), self.data.grid_data)
            np.save(os.path.join(path7, "grid_p_data"), self.data.grid_p_data)
            np.save(os.path.join(path7, "grid_mask"), self.data.grid_mask)
            np.save(os.path.join(path7, "t_data"), self.data.t_data)
            np.save(os.path.join(path7, "grid"), np.array([self.data.grid]))

    def sparsity(self, tol=1e-3):

        if self.fitted:
            sparsity = {}
            if "seas" in self.code:
                var = np.copy(self.estimate_seas)
                if tol != 0:
                    var[(np.abs(var) < tol)] = 0
                var = sum(sum((var == 0)))
                sparsity["weights_seas, size"] = self.estimate_seas.shape[0] * self.estimate_seas.shape[1]
                sparsity["weights_seas, % of non zero entries, for given tol"] = \
                    (sparsity["weights_seas, size"] - var) / sparsity["weights_seas, size"]
            if "trend" in self.code:
                var = np.copy(self.estimate_trend)
                if tol != 0:
                    var[(np.abs(var) < tol)] = 0
                var = sum(sum((var == 0)))
                sparsity["weights_trend, size"] = self.estimate_trend.shape[0] * self.estimate_trend.shape[1]
                sparsity["weights_trend, % of non zero entries, for given tol"] = \
                    (sparsity["weights_trend, size"] - var) / sparsity["weights_trend, size"]

            return sparsity

        else:
            raise Exception("Can not get sparsity of the weights as the model is not fitted.")

    def mean(self, xyz, code):

        if code == "seas":
            t = np.linspace(0, self.T, 1000)
            integral = self.field_phi_seas.evaluate(self.field_phi_seas.proj(np.clip(proj_circle(self.T, t).
                                                           dot(self.field_phi_seas.nodes.transpose()), -1, 1)))
            integral = np.mean(integral, axis=0)
            mean = self.field_psi_seas.evaluate(self.field_psi_seas.proj(np.clip(
                xyz.dot(self.field_psi_seas.nodes.transpose()), -1, 1))).dot(self.estimate_seas).dot(integral)

        if code == "trend":

            t = np.linspace(self.data.t_data[0], self.data.t_data[-1],
                            1000 * int((self.data.t_data[-1] - self.data.t_data[0]) / 12))
            integral = self.field_zeta_trend.evaluate(self.field_zeta_trend.proj(t.reshape(t.shape[0], 1) -
                                                                self.field_zeta_trend.nodes.reshape(1, self.M_trend)))
            integral = np.mean(integral, axis=0)
            mean = self.field_psi_trend.evaluate(self.field_psi_trend.proj(xyz.dot(
                self.field_psi_trend.nodes.transpose()))).dot(self.estimate_trend).dot(integral)

        return mean













