"""μFA estimation using the 2nd order signal cumulant expansion."""

import os
import numba
import numpy as np
import nibabel as nib
import scipy.optimize


@numba.jit()
def cum_model_f(x, bs):
    """Return signal based on signal model."""
    S0 = x[0]
    MD = x[1]
    V_iso = x[2]
    V_aniso = x[3]
    S_lte = S0 * np.exp(-bs * MD + .5 * bs**2 * (V_aniso + V_iso))
    S_ste = S0 * np.exp(-bs * MD + .5 * bs**2 * (V_iso))
    return np.concatenate((S_lte, S_ste))


@numba.jit()
def cum_res_f(x, pa_data, bs):
    """Return residuals."""
    res = pa_data - cum_model_f(x, bs)
    return res


def fit_cum_model(pa_lte, pa_ste, bs, x0=None,
                  bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])):
    """Fit signal model to data."""
    if x0 is None:
        x0 = np.zeros(4)
        x0[0] = np.nanmean([pa_lte[0], pa_ste[0]])  # S0 initial
        if x0[0] < 0 or np.isnan(x0[0]):
            x0[0] = np.nanmax([pa_lte, pa_ste])
        x0[1] = 1  # MD initial
        x0[2] = .1  # V_iso initial
        x0[3] = .1  # V_aniso initial
    fit = scipy.optimize.least_squares(
        fun=cum_res_f, x0=x0, args=(np.concatenate((pa_lte, pa_ste)), bs),
        bounds=bounds, method='trf')
    return fit.x


@numba.jit()
def calc_pa(data, bvals):
    """Return powder-averaged signal."""
    bs = np.unique(bvals)
    pa = np.zeros(len(bs))
    for i, b in enumerate(bs):
        pa[i] = np.nanmean(data[np.where(bvals == b)[0]])
    return pa


@numba.jit()
def calc_uFA(MD, V_aniso, V_iso):
    """Return μFA."""
    return np.sqrt(1.5 * V_aniso / (V_aniso + 0.4 * (MD**2 + V_iso)))


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
    np.seterr(all='ignore')  # Ignore expected warnings (empty slice etc.)

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
        V_iso = np.zeros((xs, ys, zs)) * np.nan
        V_aniso = np.zeros((xs, ys, zs)) * np.nan
        for i in range(xs):
            for j in range(ys):
                for k in range(zs):
                    if mask[i, j, k]:
                        pa_lte = calc_pa(data[i, j, k, lte_idx], bvals[lte_idx])
                        pa_ste = calc_pa(data[i, j, k, ste_idx], bvals[ste_idx])
                        x = fit_cum_model(pa_lte, pa_ste, bs)
                        S0[i, j, k] = x[0]
                        MD[i, j, k] = x[1]
                        V_iso[i, j, k] = x[2]
                        V_aniso[i, j, k] = x[3]
            print(str(np.round((i / xs) * 100, 0)) + ' %', end="\r")
        uFA = calc_uFA(MD, V_aniso, V_iso)
        print('100 %')

        # Save results
        if not os.path.exists(r):
            os.mkdir(r)
        nib.save(nib.Nifti1Image(S0, affine),
                 os.path.join(r, 'cum_S0.nii.gz'))
        nib.save(nib.Nifti1Image(MD, affine),
                 os.path.join(r, 'cum_MD.nii.gz'))
        nib.save(nib.Nifti1Image(V_iso, affine),
                 os.path.join(r, 'cum_V_iso.nii.gz'))
        nib.save(nib.Nifti1Image(V_aniso, affine),
                 os.path.join(r, 'cum_V_aniso.nii.gz'))
        nib.save(nib.Nifti1Image(uFA, affine),
                 os.path.join(r, 'cum_uFA.nii.gz'))
