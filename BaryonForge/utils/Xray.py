import numpy as np
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

__all__ = ['EmissivityTable',]

class EmissivityTable:
    """
    Interpolate emissivity in (T, Z, z), using log for T and emissivity.

    Parameters
    ----------
    T, Z, a: array_like
        Full list of sample-point coordinates. These must always be passed
        as point lists of equal length, even when the samples lie on a
        regular Cartesian grid. We need temperature in physical Kelvin,
        metallicity in Zsolar, and scale factor. Kelvin (not keV) is what the
        `Temperature` profiles return, and it is what `Emissivity` feeds into
        this table; use `BaryonForge.utils.K_to_kev` to convert a keV-based
        grid when you build the table.
    emissivity : array_like
        Emissivity values at the supplied points. Must have the same shape
        as T, Z after flattening. Volumes in CCL/Baryonforge are always
        in comoving units, so pass in emissivity in cm^3 --> comoving cm^3
    regular_grid : bool
        If True, interpret the input points as a full Cartesian product grid
        and use RegularGridInterpolator after validating this.
        If False, use NearestNDInterpolator on the scattered points.
    regular_grid_method : str, optional
        Interpolation method for RegularGridInterpolator. Default is "linear".
    pad_low_T : bool, optional
        Whether to pad the table so it can handle extremely low astrophysical
        temperature (T < 1e4 K) that are below the table limits. Setting this
        to True will return 0 emissivity for these regions, instead of
        returning out-of-bounds errors. These temperatures can occur in our 
        models for low-mass or large-separation scenarios.
    """

    sentinel_value = 1e-100

    def __init__(self, T, Z, a, emissivity, regular_grid=True, regular_grid_method="linear", pad_low_T = True):
        
        self.regular_grid        = regular_grid
        self.regular_grid_method = regular_grid_method
        self.pad_low_T           = pad_low_T

        T = np.asarray(T, dtype = float).ravel()
        Z = np.asarray(Z, dtype = float).ravel()
        a = np.asarray(a, dtype = float).ravel()
        emissivity = np.asarray(emissivity, dtype = float).ravel()

        assert T.size > 0, "T must not be empty."
        assert Z.size > 0, "Z must not be empty."
        assert a.size > 0, "a must not be empty"
        assert T.size == Z.size == a.size == emissivity.size, "T, Z, a, and emissivity must all have the same size."

        assert np.all(T > 0),  "All T values must be strictly positive."
        assert np.all(Z >= 0), "All Z values must be strictly non-negative."
        assert np.all(a > 0),  "All a values must be strictly positive."

        self.T_range = (T.min(), T.max())
        self.Z_range = (Z.min(), Z.max())
        self.a_range = (a.min(), a.max())

        logT = np.log(T)
        logE = np.log(emissivity + self.sentinel_value)

        if self.regular_grid:
            self.interpolator = self._build_regular_grid_interpolator(logT, Z, a, logE)
        else:
            points = np.column_stack([logT, Z, a])
            self.interpolator = NearestNDInterpolator(points, logE.ravel())


    def _build_regular_grid_interpolator(self, logT, Z, a, emissivity):
        """
        Validate that the input scattered points form a complete Cartesian
        product grid, then build a RegularGridInterpolator.
        """
        uT = np.unique(logT)
        uZ = np.unique(Z)
        ua = np.unique(a)

        nT = uT.size
        nZ = uZ.size
        na = ua.size
        n_points = logT.size
        n_expected = nT * nZ * na

        if n_expected != n_points:
            raise ValueError(
                "regular_grid=True was requested, but the input points do not "
                "contain a full Cartesian product of the unique T, Z, and a values. "
                f"Got {n_points} points, but expected "
                f"{nT} * {nZ} * {na} = {n_expected}. "
                "Please provide the entire set of grid points."
            )

        #Place every input point at its location on the (T, Z, a) grid. This way the
        #input points can be passed in any order (flat lists, or meshgrids of any indexing).
        iT = np.searchsorted(uT, logT)
        iZ = np.searchsorted(uZ, Z)
        ia = np.searchsorted(ua, a)
        values = np.full((nT, nZ, na), np.nan)
        values[iT, iZ, ia] = emissivity

        if np.any(np.isnan(values)):
            raise ValueError("regular_grid=True was requested, but some (T, Z, a) grid points are duplicated/missing.")

        return RegularGridInterpolator((uT, uZ, ua), values, method=self.regular_grid_method, bounds_error=True, fill_value=None,)


    def _assert_in_bounds(self, T, Z, a):
        assert np.all(T >= 0), "Query T values must be strictly non-negative."
        assert np.all(Z >= 0), "Query Z values must be strictly non-negative."
        assert np.all(a >= 0), "Query a values must be strictly non-negative."

        if self.pad_low_T:
            mask = (T <= self.T_range[1]); assert np.all(mask), f"Some T values ({T[~mask]}) are greater than maximum tabulated value {self.T_range[1]}."
        else:
            mask = (T >= self.T_range[0]) & (T <= self.T_range[1]); assert np.all(mask), f"Some T values ({T[~mask]}) outside tabulated range {self.T_range}."

        mask = (Z >= self.Z_range[0]) & (Z <= self.Z_range[1]); assert np.all(mask), f"Some Z values ({Z[~mask]}) outside tabulated range {self.Z_range}."
        mask = (a >= self.a_range[0]) & (a <= self.a_range[1]); assert np.all(mask), f"Some a values ({a[~mask]}) outside tabulated range {self.a_range}."



    def __call__(self, T, Z, a):

        #Flatten inputs so any shape (eg. (N_mass, N_radius)) is supported, and reshape at the end
        shape = np.shape(T)
        T = np.asarray(T, dtype = float).ravel()
        Z = np.broadcast_to(np.asarray(Z, dtype = float), shape).ravel()
        a = np.atleast_1d(a).astype(float)

        self._assert_in_bounds(T, Z, a)

        if self.pad_low_T:
            values = np.zeros_like(T)
            usage  = T >= self.T_range[0]
            points = np.column_stack([np.log(T[usage]), Z[usage], a[0] * np.ones(usage.sum())])
            tmpout = self.interpolator(points)
            values[usage] = np.exp(tmpout) + self.sentinel_value
        else:
            points = np.column_stack([np.log(T), Z, a[0] * np.ones_like(Z)])
            values = self.interpolator(points)
            values = np.exp(values) + self.sentinel_value

        return values.reshape(shape)