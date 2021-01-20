"""μFA estimation using q-space trajectory imaging."""

import os
import numba
import numpy as np
import nibabel as nib
from dipy.core.geometry import vec2vec_rotmat


@numba.jit()
def from_3x3_to_6x1(T):
    """Convert symmetric second order tensor to first order tensor."""
    C = np.sqrt(2)
    V = np.array([[T[0, 0],
                   T[1, 1],
                   T[2, 2],
                   C * T[1, 2],
                   C * T[0, 2],
                   C * T[0, 1]]]).T
    return V


@numba.jit()
def from_6x1_to_3x3(V):
    """Convert first order tensor to symmetric second order tensor."""
    C = np.sqrt(1 / 2)
    T = np.array([[V[0, 0], C * V[5, 0], C * V[4, 0]],
                  [C * V[5, 0], V[1, 0], C * V[3, 0]],
                  [C * V[4, 0], C * V[3, 0], V[2, 0]]])
    return T


@numba.jit()
def from_6x6_to_21x1(T):
    """Convert symmetric second order tensor to first order tensor."""
    C2 = np.sqrt(2)
    V = np.array([[T[0, 0], T[1, 1], T[2, 2],
                   C2 * T[1, 2], C2 * T[0, 2], C2 * T[0, 1],
                   C2 * T[0, 3], C2 * T[0, 4], C2 * T[0, 5],
                   C2 * T[1, 3], C2 * T[1, 4], C2 * T[1, 5],
                   C2 * T[2, 3], C2 * T[2, 4], C2 * T[2, 5],
                   T[3, 3], T[4, 4], T[5, 5],
                   C2 * T[3, 4], C2 * T[4, 5], C2 * T[5, 3]]]).T
    return V


@numba.jit()
def from_21x1_to_6x6(V):
    """Convert first order tensor to symmetric second order tensor."""
    v = V[:, 0]  # Code easier to read without extra dimension
    C2 = np.sqrt(1 / 2)
    T = np.array([[v[0], C2 *
                   v[5], C2 *
                   v[4], C2 *
                   v[6], C2 *
                   v[7], C2 *
                   v[8]], [C2 *
                           v[5], v[1], C2 *
                           v[3], C2 *
                           v[9], C2 *
                           v[10], C2 *
                           v[11]], [C2 *
                                    v[4], C2 *
                                    v[3], v[2], C2 *
                                    v[12], C2 *
                                    v[13], C2 *
                                    v[14]], [C2 *
                                             v[6], C2 *
                                             v[9], C2 *
                                             v[12], v[15], C2 *
                                             v[18], C2 *
                                             v[20]], [C2 *
                                                      v[7], C2 *
                                                      v[10], C2 *
                                                      v[13], C2 *
                                                      v[18], v[16], C2 *
                                                      v[19]], [C2 *
                                                               v[8], C2 *
                                                               v[11], C2 *
                                                               v[14], C2 *
                                                               v[20], C2 *
                                                               v[19], v[17]]])
    return T


E_iso = np.eye(6) / 3
E_bulk = np.dot(from_3x3_to_6x1(E_iso), from_3x3_to_6x1(E_iso).T)
E_shear = E_iso - E_bulk
spherical_tensor = np.eye(3) / 3
linear_tensor = np.zeros((3, 3))
linear_tensor[0, 0] = 1


def calc_design_matrix(lte_bvecs, lte_bvals, ste_bvecs, ste_bvals):
    """Return design matrix. The order of acquisitions is hardcoded."""
    N = len(lte_bvals) + len(ste_bvals)
    btens = np.zeros((N, 3, 3))
    for i, b in enumerate(lte_bvals):
        R = vec2vec_rotmat(np.array([1, 0, 0]), lte_bvecs[i])
        btens[i] = np.matmul(np.matmul(R, linear_tensor), R.T) * b
    for i, b in enumerate(ste_bvals):
        btens[len(lte_bvals) + i] = spherical_tensor * b
    X = np.zeros((N, 28))
    for i in range(N):
        b = from_3x3_to_6x1(btens[i])
        b_sq = from_6x6_to_21x1(np.matmul(b, b.T))
        X[i, :] = np.concatenate(([1], (-b.T)[0, :], (0.5 * b_sq.T)[0, :]))
    return X


def fit_qti(S, X):
    """Fit QTI to data."""
    N = len(S)
    C = np.eye(N) * S**2
    S = np.log(S)[:, np.newaxis]
    S[np.isnan(S)] = 0  # In case signal <= 0
    A = np.matmul(np.matmul(X.T, C), X)
    B = np.matmul(X.T, C)
    beta = np.matmul(np.matmul(np.linalg.pinv(A), B), S)
    S0 = beta[0]
    D = beta[1:7]
    C = beta[7::]
    return S0, D, C


def calc_uFA(D, C):
    """Return μFA."""
    meanD_sq = np.matmul(D, D.T)
    mean_Dsq = from_21x1_to_6x6(C) + meanD_sq
    uFA = np.sqrt(1.5 * (np.matmul(from_6x6_to_21x1(mean_Dsq).T,
                                   from_6x6_to_21x1(E_shear))) /
                  np.matmul(from_6x6_to_21x1(mean_Dsq).T,
                            from_6x6_to_21x1(E_iso)))
    return uFA


if __name__ == '__main__':  # Fit to phantom and volunteer data

    CODE_DIR = os.getcwd()
    BASE_DIR = os.path.dirname(CODE_DIR)
    DATA_DIR = os.path.join(BASE_DIR, 'Preproc-data')
    RESULTS_DIR = os.path.join(BASE_DIR, 'Results')
    SUBJECT_DIRS = sorted(
        [os.path.join(DATA_DIR, i) for i in os.listdir(DATA_DIR) if
         os.path.isdir(os.path.join(DATA_DIR, i))])
    RESULTS_DIRS = sorted(
        [os.path.join(RESULTS_DIR, i) for i in os.listdir(DATA_DIR) if
         os.path.isdir(os.path.join(DATA_DIR, i))])
    lte_idx = np.arange(107)
    ste_idx = np.arange(107, 214)
    np.seterr(all='ignore')  # Ignore expected runtime warnings

    for s, r in zip(SUBJECT_DIRS, RESULTS_DIRS):

        print('Fitting model to data at %s' % s)

        # Load data
        data = nib.load(os.path.join(s, 'LTE-STE.nii.gz')).get_fdata()
        xs, ys, zs, N = data.shape
        affine = nib.load(os.path.join(s, 'LTE-STE.nii.gz')).affine
        bvals = np.loadtxt(os.path.join(s, 'LTE-STE.bval')) * 1e-3
        bs = np.unique(bvals)
        bvecs = np.loadtxt(os.path.join(s, 'LTE-STE.bvec')).T
        mask = nib.load(os.path.join(s, 'mask.nii.gz')).get_fdata().astype(bool)

        # Mask data
        data[~mask] = np.nan

        # Fit model by looping over voxel (slow)
        S0 = np.zeros((xs, ys, zs)) * np.nan
        MD = np.zeros((xs, ys, zs)) * np.nan
        uFA = np.zeros((xs, ys, zs)) * np.nan
        X = calc_design_matrix(
            bvecs[lte_idx], bvals[lte_idx], bvecs[ste_idx], bvals[ste_idx])
        for i in range(xs):
            for j in range(ys):
                for k in range(zs):
                    if mask[i, j, k]:
                        S0_hat, D_hat, C_hat = fit_qti(data[i, j, k], X)
                        S0[i, j, k] = S0_hat
                        MD[i, j, k] = np.trace(from_6x1_to_3x3(D_hat)) / 3
                        uFA[i, j, k] = calc_uFA(D_hat, C_hat)
            print(str(np.round((i / xs) * 100, 0)) + ' %', end="\r")
        print('100 %')

        # Save results
        if not os.path.exists(r):
            os.mkdir(r)
        nib.save(nib.Nifti1Image(S0, affine),
                 os.path.join(r, 'qti_S0.nii.gz'))
        nib.save(nib.Nifti1Image(MD, affine),
                 os.path.join(r, 'qti_MD.nii.gz'))
        nib.save(nib.Nifti1Image(uFA, affine),
                 os.path.join(r, 'qti_uFA.nii.gz'))
