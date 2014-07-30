import SphericalGrid
import shtns
import numpy as np
import scipy

def truncate_tiny(matrix):
    norm = np.sqrt(np.sum(matrix**2))
    matrix[np.abs(matrix/norm) < 1e-15] = 0.0


class ConformalAKVSol:
    def __init__(self, psi_coeffs, resolution = 15, nsols = 3, min_norm = 'Killing', grid = None):
        if grid == None:
            grid = SphericalGrid.SphericalGrid(resolution, resolution)

        psi = grid.SpecToPhys(psi_coeffs)

        grid.gthth, grid.gphph = np.exp(2*psi), np.exp(2*psi)*grid.sintheta**2
        grid.UpdateMetric()
        grid.ComputeMetricDerivs()

        grid.ricci = 2/grid.gthth*(1 - grid.SphereLaplacian(psi))

        f1, f2 = np.ones(grid.extents), np.zeros(grid.extents)
        zeros = np.zeros(grid.numTerms)
        dR = grid.D(grid.ricci)
        gradR = grid.Raise(dR)

        N = grid.numTerms - 1

        if nsols > N:
            nsols = N
        self.nsols = nsols

        if min_norm == 'Killing':
            A = np.empty((N,N))
            E = np.empty((N,N))
            B = np.empty((N,N))
            L = np.empty((N,N))

            for i in xrange(N):
                coeffs = np.zeros(grid.numTerms)
                coeffs[i+1] = 1.0
                f = grid.SpecToPhys(coeffs)
                df = grid.D(f)

                Lf = grid.Laplacian(f)
                LLf = grid.Laplacian(Lf)

                RLf = grid.ricci*Lf

                gradRdf = gradR[0]*df[0] + gradR[1]*df[1]

                epsdRdf = (dR[0]*df[1] - dR[1]*df[0])/grid.dA

                Af = RLf + gradRdf + 2*LLf

                Bf = RLf + gradRdf + LLf

                A[:,i] = grid.PhysToSpec(Af)[1:]
                E[:,i] = grid.PhysToSpec(epsdRdf)[1:]
                B[:,i] = grid.PhysToSpec(Bf)[1:]
                L[:,i] = grid.PhysToSpec(Lf)[1:]

            for m in A, E, B, L:
                truncate_tiny(m)

            M1 = np.bmat([[A,E],[-E,B]])
            M2 = np.bmat([[L,0*L],[0*L,L]])

            eigenvals, v = scipy.linalg.eig(M1, M2)

            v = v.T

            truncate_tiny(v)

            v1, v2 = v[:,:N], v[:,N:]

        else:
            H = np.empty((N,N))
            L = np.empty((N,N))

            for i in xrange(N):
                coeffs = np.zeros(grid.numTerms)
                coeffs[i+1] = 1.0
                f = grid.SpecToPhys(coeffs)
                df = grid.D(f)

                Lf = grid.Laplacian(f)
                LLf = grid.Laplacian(Lf)

                RLf = grid.ricci*Lf

                gradRdf = gradR[0]*df[0] + gradR[1]*df[1]

                Hf = RLf + gradRdf + LLf

                H[:,i] = grid.PhysToSpec(Hf)[1:]
                L[:,i] = grid.PhysToSpec(Lf)[1:]

            truncate_tiny(H)
            truncate_tiny(L)

            eigenvals, v = scipy.linalg.eig(H, L)

            v = v.T
            truncate_tiny(v)

            v1, v2 = 0*v, v
            
        v1 = np.column_stack((np.zeros(len(v1)), v1))
        v2 = np.column_stack((np.zeros(len(v2)), v2))
        
        f1, f2 = np.array([grid.SpecToPhys(v) for v in v1]), np.array([grid.SpecToPhys(v) for v in v2])
        df1, df2 = np.array([grid.D(F1) for F1 in f1]), np.array([grid.D(F2) for F2 in f2])
        
        norm = np.array([grid.Integrate(-F2*grid.Laplacian(F2) - F1*grid.Laplacian(F1) + 2*(DF1[0]*DF2[1] - DF1[1]*DF2[0])/grid.dA) for F1,F2, DF1, DF2 in zip(f1,f2, df1, df2)])

        f1 = np.array([F1/np.sqrt(N) for F1, N in zip(f1, norm)])
        f2 = np.array([F2/np.sqrt(N) for F2, N in zip(f2, norm)])

        knorms = -0.5*eigenvals.real

        sorted_index = np.abs(knorms).argsort()

        eigenvals, v1,  v2, knorms,  df1, df2 = eigenvals[sorted_index], v1[sorted_index], v2[sorted_index], knorms[sorted_index], df1[sorted_index], df2[sorted_index]

        self.vector_fields = np.array([np.array([DF2[1], -DF2[0]])/grid.dA + DF1 for DF1, DF2 in zip(df1[:nsols], df2[:nsols])])
        self.knorms = knorms[:nsols]

        self.knorms[np.abs(self.knorms) < 1e-15] = 0.0

        self.eigenvals = eigenvals[:nsols]
        
        self.f1, self.f2 = f1[:nsols], f2[:nsols]

        self.grid = grid

    def GetVectorFields(self):
        return self.vector_fields

    def GetEigenvalues(self):
        return self.eigenvals
        
    def GetKillingNorms(self):
        return self.knorms

    def GetFunctions(self):
        return self.f1, self.f2

    def GetGrid(self):
        return self.grid
