import numpy as np
import healpy as hp
import warnings

__all__ = ['HaloLightConeCatalog', 'HaloNDCatalog', 'LightconeShell', 
           'GriddedMap', 'ParticleSnapshot']


class HaloLightConeCatalog(object):
    """
    A class to read and store a halo catalog in a lightcone, along with cosmological parameters.

    The `HaloLightConeCatalog` class is used in the baryonification pipeline to manage data related to halos
    observed in a lightcone, including their right ascension (RA), declination (Dec), mass (M), and redshift (z).
    The class also stores additional quantities as provided and ensures that necessary cosmological parameters
    are specified.

    Parameters
    ----------
    ra : array-like
        The right ascension values of the halos. Can be a list, numpy array, or tuple.
    
    dec : array-like
        The declination values of the halos. Can be a list, numpy array, or tuple. Declination values 
        exactly at the poles (±90 degrees) are slightly offset (by 1e-5 arcsec) to avoid singularities.
    
    M : array-like
        The mass values of the halos. Can be a list, numpy array, or tuple.
    
    z : array-like
        The redshift values of the halos. Can be a list, numpy array, or tuple.
    
    cosmo : dict
        A dictionary containing the cosmological parameters. Must include keys for 'Omega_m', 'sigma8', 'h',
        'Omega_b', 'n_s', and 'w0'.
    
    **arrays : a kwargs dictionary
        Additional arrays to be included in the catalog. Each key-value pair corresponds to a name and an array
        of values of that parameter for the halos.

    Attributes
    ----------
    cat : structured ndarray
        A structured numpy array containing the halo data, including RA, Dec, mass, redshift, and any additional
        quantities provided.
    
    cosmo : dict
        The cosmological parameters provided at initialization.

    Raises
    ------
    ValueError
        If not all required cosmological parameters are provided.
    """

    def __init__(self, ra, dec, M, z, cosmo, **arrays):

        t     = np.float64
        dtype = [('M', t), ('z', t), ('ra', t), ('dec', t)]
        dtype = dtype + [(name, t) for name, arr in arrays.items()]
            
        N   = 1 if not isinstance(ra, (list, np.ndarray, tuple)) else len(ra)
        cat = np.zeros(len(ra), dtype)

        if np.any(np.abs(dec) == 90):
            dec = dec.astype(t) #Need to upgrade type so the subtraction below is still accurate
            warnings.warn("Some halos found with declination exactly at the poles. Offsetting these by 4e-5 arcsec")
            dec = np.clip(dec, -90 + 1e-8, 90 - 1e-8)
        
        cat['ra']  = ra
        cat['dec'] = dec
        cat['z']   = z
        cat['M']   = M
        
        for name, arr in arrays.items(): cat[name] = arr

        self.cat   = cat

        keys = cosmo.keys()
        if not (('Omega_m' in keys) & ('sigma8' in keys) & ('h' in keys) &
                ('Omega_b' in keys) & ('n_s' in keys) & ('w0' in keys)):

            raise ValueError("Not all cosmology parameters provided. I need Omega_m, sigma8, h, sigma8, Omega_b, n_s, w0")
        else:
            self.cosmo = cosmo

    @property
    def data(self):
        """
        Returns the structured array containing halo data.
        """

        return self.cat

    @property
    def cosmology(self):
        """
        Returns the cosmology dictionary.
        """

        return self.cosmo
    
    
    def __getitem__(self, key):
        """
        Returns a new `HaloLightConeCatalog` object with halos corresponding to the specified key.

        Parameters
        ----------
        key : int or slice
            Index or slice to select specific halos from the catalog.

        Returns
        -------
        HaloLightConeCatalog
            A new `HaloLightConeCatalog` instance with the selected subset of halos.
        """
        
        ra  = self.cat['ra'][key]
        dec = self.cat['dec'][key]
        z   = self.cat['z'][key]
        M   = self.cat['M'][key]
                        
        other = {}
        for k in self.cat.dtype.names:
            if k not in ['ra', 'dec', 'M', 'z']:
                other[k] = self.cat[k][key]
        
        return HaloLightConeCatalog(ra = ra, dec = dec, M = M, z = z, cosmo = self.cosmo, **other)

    
    def __str__(self):
        
        string = f"""
HaloLightConeCatalog with {self.cat.size} Halos at {self.cat['z'].min()} < z < {self.cat['z'].max()}.
Minimum log10(Mass) = {np.log10(self.cat['M'].min())}
Maximum log10(Mass) = {np.log10(self.cat['M'].max())}
Cosmology set to {self.cosmo}.
        """
        return string.strip()
    

class HaloNDCatalog(object):
    """
    A class to read and store a halo catalog in a 2D or 3D field, along with cosmological parameters.

    The `HaloNDCatalog` class is used in the baryonification pipeline to manage data related to halos in 
    either a 2D or 3D field. The catalog includes information about halos' positions, masses, and any other
    provided attributes, as well as the cosmological parameters necessary for analysis.

    Parameters
    ----------
    x : array-like
        The x-coordinates of the halos. Can be a list, numpy array, or tuple.
    
    y : array-like
        The y-coordinates of the halos. Can be a list, numpy array, or tuple.
    
    M : array-like
        The mass values of the halos. Can be a list, numpy array, or tuple.
    
    redshift : float
        The redshift value associated with the halos.
    
    cosmo : dict
        A dictionary containing the cosmological parameters. Must include keys for 'Omega_m', 'sigma8', 'h',
        'Omega_b', 'n_s', and 'w0'.
    
    z : array-like, optional
        The z-coordinates of the halos. Default is None, which sets the z-coordinates to 0, indicating a 2D field.
    
    **arrays : a kwargs dict
        Additional arrays to be included in the catalog. Each key-value pair corresponds to a name and an array
        of values of that parameter for the halos.

    Attributes
    ----------
    cat : structured ndarray
        A structured numpy array containing the halo data, including x, y, z (if applicable), mass, and any additional
        quantities provided.
    
    redshift : float
        The redshift value associated with the halos.
    
    cosmo : dict
        The cosmological parameters provided at initialization.

    Methods
    -------
    data
        Returns the structured array containing halo data.
    
    cosmology
        Returns the dictionary of cosmological parameters.
    
    Raises
    ------
    ValueError
        If not all required cosmological parameters are provided.
    """

    def __init__(self, x, y, M, redshift, cosmo, z = None, **arrays):

        dtype = [('M', '>f'), ('x', '>f'), ('y', '>f'), ('z', '>f')]
        dtype = dtype + [(name, '>f', arr.shape[1:] if len(arr.shape) > 1 else '') for name, arr in arrays.items()]
        
        
        N = 1 if not isinstance(x, (list, np.ndarray, tuple)) else len(x)
        cat = np.zeros(N, dtype)

        cat['x'] = x
        cat['y'] = y
        cat['z'] = 0 if z is None else z #We'll just add filler to z-column for now
        cat['M'] = M
        
        for name, arr in arrays.items(): cat[name] = arr

        self.cat = cat
        self.redshift = redshift

        keys = cosmo.keys()
        if not (('Omega_m' in keys) & ('sigma8' in keys) & ('h' in keys) &
                ('Omega_b' in keys) & ('n_s' in keys) & ('w0' in keys)):

            raise ValueError("Not all cosmology parameters provided. I need Omega_m, sigma8, h, sigma8, Omega_b, n_s, w0")
        else:
            self.cosmo = cosmo

    @property
    def data(self):
        """
        Returns the structured array containing halo data.
        """

        return self.cat

    @property
    def cosmology(self):
        """
        Returns the dictionary of cosmological parameters.
        """

        return self.cosmo
    
    
    def __getitem__(self, key):
        """
        Returns a new `HaloNDCatalog` object with halos corresponding to the specified key.

        Parameters
        ----------
        key : int or slice
            Index or slice to select specific halos from the catalog.

        Returns
        -------
        HaloNDCatalog
            A new `HaloNDCatalog` instance with the selected subset of halos.
        """
        
        x = self.cat['x'][key]
        y = self.cat['y'][key]
        z = self.cat['z'][key]
        M = self.cat['M'][key]
        
        other = {}
        for k in self.cat.dtype.names:
            if k not in ['x', 'y', 'z', 'M']:
                other[k] = self.cat[k][key]
        
        return HaloNDCatalog(x = x, y = y, z = z, M = M, redshift = self.redshift, cosmo = self.cosmo, **other)
    
    
    def __str__(self):
        
        string = f"""
HaloNDCatalog with {self.cat.size} Halos at z = {self.redshift}.
Minimum log10(Mass) = {np.log10(self.cat['M'].min())}
Maximum log10(Mass) = {np.log10(self.cat['M'].max())}
Cosmology set to {self.cosmo}.
        """
        return string.strip()
    

    def __repr__(self):
        
        return f"HaloNDCatalog(cat = {self.cat!r}, \nredshift = {self.redshift!r}, \ncosmo = {self.cosmo})"


class LightconeShell(object):

    """
    A class to read and store a lightcone shell (HEALPix map) along with cosmological parameters.

    The `LightconeShell` class is used in the pipeline to manage data related to a lightcone shell,
    which is represented as a HEALPix map. This class facilitates the reading and storage of the map data, either
    directly from an array or by loading from a file, and ensures that necessary cosmological parameters are specified.

    Parameters
    ----------
    map : ndarray, optional
        A numpy array containing the HEALPix map data. The map must be in ring configuration.
        If not provided, `path` must be specified.

    path : str, optional
        A string representing the path to a file containing the HEALPix map data. If not provided,
        `map` must be specified.

    cosmo : dict, optional
        A dictionary containing the cosmological parameters. Must include keys for 'Omega_m', 'sigma8', 'h',
        'Omega_b', 'n_s', and 'w0'.

    redshift : float
        The redshift value associated with the map.

    Attributes
    ----------
    map : ndarray
        A numpy array representing the HEALPix map data.

    NSIDE : int
        The NSIDE parameter of the HEALPix map, calculated from the map size.

    cosmo : dict
        The cosmological parameters provided at initialization.

    Methods
    -------
    data
        Returns the HEALPix map data.
    
    cosmology
        Returns the dictionary of cosmological parameters.

    Raises
    ------
    ValueError
        If neither `map` nor `path` is provided, or if the required cosmological parameters are not included.
    """

    def __init__(self, map = None, path = None, cosmo = None, redshift = None):

        if (path is None) & (map is None):
            raise ValueError("Need to provide either path to map, or provide map values in healpix ring configuration")

        elif isinstance(path, str):
            self.map = hp.read_map(path)

        elif isinstance(map, np.ndarray):
            self.map = map


        self.NSIDE    = hp.npix2nside(self.map.size)
        self.redshift = redshift

        
        keys = cosmo.keys()
        if not (('Omega_m' in keys) & ('sigma8' in keys) & ('h' in keys) &
                ('Omega_b' in keys) & ('n_s' in keys) & ('w0' in keys)):

            raise ValueError("Not all cosmology parameters provided. I need Omega_m, sigma8, h, sigma8, Omega_b, n_s, w0")
        else:
            self.cosmo = cosmo

    @property
    def data(self):
        """
        Returns the HEALPix map data.
        """

        return self.map

    @property
    def cosmology(self):
        """
        Returns the dictionary of cosmological parameters.
        """

        return self.cosmo
    

class GriddedMap(object):

    """
    A class to read and store a gridded map (either 2D or 3D) along with cosmological parameters.

    The `GriddedMap` class is used in the baryonification pipeline to manage data related to a gridded map,
    which can be either a 2D or 3D representation of a field. This class facilitates the storage of the map
    data, resolution, and bin information, and ensures that necessary cosmological parameters are specified.

    Parameters
    ----------
    map : ndarray
        A numpy array containing the gridded map data. Can be either 2D or 3D.
    
    redshift : float
        The redshift value associated with the map.
    
    bins : ndarray
        A numpy array representing the coordinates of the map along a given axis, in physical Mpc. It determines
        the spacing and extent of the grid. This is the center of each pixel.
    
    cosmo : dict
        A dictionary containing the cosmological parameters. Must include keys for 'Omega_m', 'sigma8', 'h',
        'Omega_b', 'n_s', and 'w0'.

    Attributes
    ----------
    map : ndarray
        A numpy array representing the gridded map data.
    
    redshift : float
        The redshift value associated with the map.
    
    Npix : int
        The number of pixels along one axis of the map (assuming the map is square for 2D or cubic for 3D).
    
    res : float
        The resolution of the grid, calculated as the difference between consecutive bin values.
    
    bins : ndarray
        The bin coordinates of the map, in physical Mpc. This is the center of each pixel.
    
    is2D : bool
        A boolean indicating whether the map is 2D (`True`) or 3D (`False`).
    
    grid : list of ndarrays
        A list of numpy arrays representing the meshgrid of bin coordinates, useful for indexing.
    
    inds : ndarray
        An array of indices corresponding to positions in the grid.
    
    cosmo : dict
        The cosmological parameters provided at initialization.

    Methods
    -------
    data
        Returns the gridded map data.
    
    cosmology
        Returns the dictionary of cosmological parameters.

    Raises
    ------
    ValueError
        If the map is not square or cubic, or if the required cosmological parameters are not included.
    """

    def __init__(self, map = None, redshift = None, bins = None, cosmo = None):
        
        self.map      = map
        self.redshift = redshift
        self.Npix     = self.map.shape[0]
        self.res      = bins[1] - bins[0]
        self.bins     = bins
        self.L        = bins[-1] + self.res/2 #Box size is center of last bin + 1/2 the bin width
        
        self.is2D = True if len(self.map.shape) == 2 else False

        if self.is2D:
            assert self.map.shape[0] == self.map.shape[1] #Maps have to be square maps
            self.grid = np.meshgrid(bins, bins, indexing = 'xy')
        else:
            assert (self.map.shape[0] == self.map.shape[1]) & (self.map.shape[1] == self.map.shape[2]) #Maps have to be cubic maps
            self.grid = np.meshgrid(bins, bins, bins, indexing = 'xy')

        assert self.Npix == self.bins.size, f"Map has {self.Npix} pixels a side, but you passed {self.bins.size} bins"
            
        self.inds = np.arange(self.grid[0].size).reshape(self.grid[0].shape)

        keys = cosmo.keys()
        if not (('Omega_m' in keys) & ('sigma8' in keys) & ('h' in keys) &
                ('Omega_b' in keys) & ('n_s' in keys) & ('w0' in keys)):

            raise ValueError("Not all cosmology parameters provided. I need Omega_m, sigma8, h, sigma8, Omega_b, n_s, w0")
        else:
            self.cosmo = cosmo

    @property
    def data(self):
        """
        Returns the gridded map data.
        """

        return self.map

    @property
    def cosmology(self):
        """
        Returns the dictionary of cosmological parameters.
        """

        return self.cosmo


class ParticleSnapshot(object):
    """
    A class for handling particle snapshots used in the baryonification pipeline.

    The `ParticleSnapshot` class manages data from a snapshot of particles in a simulation. It stores particle 
    positions and masses, along with relevant cosmological information. The class provides functionality to generate 
    2D or 3D gridded maps of mass distribution from the particle data.

    Parameters
    ----------
    x : array_like, optional
        The x-coordinates of the particles.
    
    y : array_like, optional
        The y-coordinates of the particles.
    
    z : array_like, optional
        The z-coordinates of the particles. If `None`, the snapshot is assumed to be 2D.
    
    M : array_like, optional
        The masses of the particles.
    
    L : float, optional
        The size of the simulation box in comoving Mpc. This determines the extent of the grid.
    
    redshift : float, optional
        The redshift corresponding to the snapshot. This is used for associating the snapshot with a specific
        cosmological epoch.
    
    cosmo : dict, optional
        A dictionary containing cosmological parameters required for baryonification. The required parameters are
        `Omega_m`, `sigma8`, `h`, `Omega_b`, `n_s`, and `w0`.

    Attributes
    ----------
    L : float
        The size of the simulation box, in comoving Mpc.
    
    cat : ndarray
        A structured NumPy array containing particle data, with columns for mass ('M') and coordinates ('x', 'y', 'z').
    
    redshift : float
        The redshift corresponding to the snapshot.
    
    is2D : bool
        A boolean flag indicating whether the snapshot is 2D (`True`) or 3D (`False`).
    
    cosmo : dict
        A dictionary containing the cosmological parameters.

    Methods
    -------
    data
        Returns the particle data as a structured array.
    
    cosmology
        Returns the dictionary of cosmological parameters.
    
    make_map(N_grid)
        Generates a 2D or 3D map from the snapshot data, based on the specified grid resolution.

    Examples
    --------
    >>> x = np.random.rand(1000) * 100
    >>> y = np.random.rand(1000) * 100
    >>> z = np.random.rand(1000) * 100
    >>> M = np.random.rand(1000) * 1e10
    >>> L = 100.0
    >>> redshift = 0.5
    >>> cosmo = {
    ...     'Omega_m': 0.3,
    ...     'sigma8': 0.8,
    ...     'h': 0.7,
    ...     'Omega_b': 0.045,
    ...     'n_s': 0.96,
    ...     'w0': -1.0
    ... }
    >>> snapshot = ParticleSnapshot(x=x, y=y, z=z, M=M, L=L, redshift=redshift, cosmo=cosmo)
    >>> map_2d = snapshot.make_map(N_grid=100)

    Notes
    -----
    - The input `cosmo` dictionary must contain all the required cosmological parameters: `Omega_m`, `sigma8`, `h`,
      `Omega_b`, `n_s`, and `w0`. Otherwise, a `ValueError` is raised.
    - If `z` is not provided, the snapshot is assumed to be 2D, and `is2D` is set to `True`.
    - The `make_map` function uses the `np.histogramdd` function to bin particle positions into a grid,
      weighting by their mass to create the map.
    """

    def __init__(self, x = None, y = None, z = None, M = None, L = None, redshift = None, cosmo = None):
        
        dtype = [('M', np.float64), ('x', np.float64), ('y', np.float64), ('z', np.float64)]
        
        cat = np.zeros(len(x), dtype)

        cat['x'] = x
        cat['y'] = y
        cat['z'] = 0 if z is None else z #We'll just add filler to z-column for now
        cat['M'] = M

        self.L   = L
        self.cat = cat
        self.redshift = redshift

        self.is2D = True if z is None else False

        keys = cosmo.keys()
        if not (('Omega_m' in keys) & ('sigma8' in keys) & ('h' in keys) &
                ('Omega_b' in keys) & ('n_s' in keys) & ('w0' in keys)):

            raise ValueError("Not all cosmology parameters provided. I need Omega_m, sigma8, h, sigma8, Omega_b, n_s, w0")
        else:
            self.cosmo = cosmo


    @property
    def data(self):
        """
        Returns the gridded map data.
        """

        return self.cat

    @property
    def cosmology(self):
        """
        Returns the dictionary of cosmological parameters.
        """

        return self.cosmo
    
    
    def make_map(self, N_grid):
        """
        Generates a 2D or 3D map from the snapshot data.

        The `make_map` function creates a gridded map (2D or 3D) from the halo catalog by binning halo positions
        into a grid of specified resolution. The halos are weighted by their mass to produce the final map. This 
        function is useful for visualizing the mass distribution in a simulated field or analyzing the density 
        field for further baryonification processing.

        Parameters
        ----------
        N_grid : int
            The number of grid cells along each axis. This determines the resolution of the resulting map.

        Returns
        -------
        Map : ndarray
            A 2D or 3D numpy array representing the gridded map of mass distribution. The dimensionality of the
            map corresponds to the dimensionality of the input catalog (2D or 3D).

        Raises
        ------
        AssertionError
            If the mass values ('M') in the catalog contain NaNs, indicating that the particle mass is not provided.

        Notes
        -----
        - The function assumes that the `cat` attribute contains 'x', 'y', and optionally 'z' positions of halos, 
        as well as their masses ('M').
        - The size of the simulation box is assumed to be `self.L`.
        - The resulting map will have shape `(N_grid, N_grid)` for 2D maps or `(N_grid, N_grid, N_grid)` for 3D maps.
        """

        assert np.isnan(self.cat['M']).sum() == 0, "If you want to make a map, provide a value for the particle mass"
        
        bins = np.linspace(0, self.L, N_grid + 1)
        
        if self.is2D:
            coords = np.vstack([self.cat['x'], self.cat['y']]).T
            bins   = (bins, bins)
            
        else:
            coords = np.vstack([self.cat['x'], self.cat['y'], self.cat['z']]).T
            bins   = (bins, bins, bins)
            

        Map = np.histogramdd(coords, bins = bins, weights = self.cat['M'])[0]

        return Map
