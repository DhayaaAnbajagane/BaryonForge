import numpy as np
import pyccl as ccl
from scipy.spatial import KDTree 
from tqdm import tqdm
from ..utils import ParamTabulatedProfile
from ..utils.Tabulate import _get_parameter
from ..Profiles.BaryonCorrection import BaryonificationClass

__all__ = ['DefaultRunnerSnapshot', 'BaryonifySnapshot']

class DefaultRunnerSnapshot(object):
    """
    A utility class for handling input/output operations related to HaloNDCatalogs and particle snapshots.

    The `DefaultRunnerSnapshot` class provides methods to manage and process data associated with halo ND catalogs
    and particle snapshots, including distance calculations with periodic boundary conditions. It uses a KDTree 
    for efficient nearest-neighbor searches within the particle snapshot.

    The initialization of this class builds a `scipy.spatial.KDTree` object to use in querying particles around
    a given halo. This step takes some considerable time.

    Parameters
    ----------
    HaloNDCatalog : object
        An instance of a `HaloNDCatalog`, containing data about halos and their properties. It must have a 
        `cosmology` attribute to specify the cosmological parameters.
    
    ParticleSnapshot : object
        An instance representing a `ParticleSnapshot` containing the positions of particles
        in the simulation.
    
    epsilon_max : float
        A parameter specifying the maximum size, in units of halo radius, of cutouts made around
        each halo during painting/baryonification.
    
    model : object, optional
        An object that generates profiles or displacements. For example, see `Baryonification2D` or `Pressure`
    
    mass_def : object, optional
        An instance of a mass definition object from the CCL (Core Cosmology Library), specifying the 
        mass definition to be used. Default is `ccl.halos.massdef.MassDef(200, 'critical')`.
    
    verbose : bool, optional
        A flag to enable verbose output for logging or debugging purposes. Default is True.

    Attributes
    ----------
    HaloNDCatalog : object
        The halo ND catalog instance.
    
    ParticleSnapshot : object
        The particle snapshot instance.
    
    cosmo : object
        The cosmology object extracted from `HaloNDCatalog`.
    
    model : object
        The model used for baryonification or profile painting.

    epsilon_max : float
        The maximum radius, in halo radius units, of cutouts around halos.
    
    mass_def : object
        The mass definition object.
    
    verbose : bool
        Whether verbose output is enabled.
    
    tree : KDTree
        A KDTree built from the particle coordinates for efficient nearest-neighbor searches.

    Methods
    -------
    compute_distance(*args)
        Helper function that computes the Euclidean distance between points, 
        accounting for periodic boundary conditions.

    enforce_periodicity(dx)
        Helper function that adjusts distances to enforce periodic boundary conditions,
        ensuring distances are within the box size.
    """
    
    def __init__(self, HaloNDCatalog, ParticleSnapshot, epsilon_max, model,
                 mass_def = ccl.halos.massdef.MassDef(200, 'critical'), verbose = True, KDTree_kwargs = {}):

        self.HaloNDCatalog    = HaloNDCatalog
        self.ParticleSnapshot = ParticleSnapshot
        self.epsilon_max      = epsilon_max
        self.cosmo = HaloNDCatalog.cosmology
        self.model = model
        
        self.mass_def = mass_def
        self.verbose  = verbose
        
        if ParticleSnapshot.is2D:
            coords = np.vstack([ParticleSnapshot.cat['x'], ParticleSnapshot.cat['y']]).T
        else:
            coords = np.vstack([ParticleSnapshot.cat['x'], ParticleSnapshot.cat['y'], ParticleSnapshot.cat['z']]).T
                               
        self.tree = KDTree(coords, boxsize = ParticleSnapshot.L, **KDTree_kwargs)

                
    def compute_distance(self, *args):
        """
        Computes the Euclidean distance between points, accounting for periodic boundary conditions.

        This method calculates the distance between points, ensuring that the computed distance takes into account
        the periodicity of the simulation box. It is designed to handle cases where distances might wrap around
        the edges of the box.

        Parameters
        ----------
        *args : list of ndarrays
            Arrays representing differences in each dimension (e.g., dx, dy, dz) between the points.

        Returns
        -------
        d : ndarray
            An array of distances computed for each pair of points, with periodicity accounted for.
        """
        
        L = self.ParticleSnapshot.L
        d = 0
        
        for dx in args:
            
            dx = np.where(dx > L/2,  dx - L, dx)
            dx = np.where(dx < -L/2, dx + L, dx)
            
            d += dx**2
            
        return np.sqrt(d)
    
    
    def enforce_periodicity(self, dx):
        """
        Adjusts distances to enforce periodic boundary conditions.

        This method adjusts the input distances to ensure that they are within the box size, effectively
        enforcing periodic boundary conditions. It modifies the distances in place.

        Parameters
        ----------
        dx : ndarray
            An array of distances to be adjusted for periodic boundary conditions.

        Returns
        -------
        dx : ndarray
            The adjusted distances, with values wrapped around the box size if necessary.
        """
        
        L = self.ParticleSnapshot.L
        
        dx = np.where(dx > L/2,  dx - L, dx)
        dx = np.where(dx < -L/2, dx + L, dx)
            
        return dx
    
    

class BaryonifySnapshot(DefaultRunnerSnapshot):
    """
    A class to apply baryonification to a particle snapshot using a halo catalog.

    The `BaryonifySnapshot` class inherits from `DefaultRunnerSnapshot` and is designed to process a particle
    snapshot by applying baryonification techniques to adjust particle positions. It uses a halo catalog to
    determine the necessary adjustments based on halo properties, cosmological parameters, and a specified model.

    Methods
    -------
    process()
        Processes the particle snapshot by applying baryonification and returns the modified particle catalog.
    """

    def process(self):
        """
        Applies baryonification to the particle snapshot using the halo catalog.

        This method iterates over each halo in the `HaloNDCatalog`, calculating the necessary displacements
        for particles within a certain radius of each halo. The displacements are computed based on the halo's
        mass, position, and scale factor. The resulting offsets are applied to the particle positions, and the
        modified particle catalog is returned.

        Returns
        -------
        new_cat : ndarray
            A structured array representing the modified particle catalog after baryonification.

        Notes
        -----
        - This method supports both 2D and 3D particle snapshots.
        - The KDTree is used for efficient querying of particles within the radius of influence of each halo.
        - Periodic boundary conditions are enforced to ensure particles remain within the simulation box.
        - The method assumes that the input catalog provides particle coordinates as 'x', 'y', and optionally 'z'.
        """

        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        L = self.ParticleSnapshot.L
        is2D        = self.ParticleSnapshot.is2D
        tot_offsets = np.zeros([len(self.ParticleSnapshot.cat), 2 if is2D else 3])

        keys = vars(self.model).get('p_keys', []) #Check if model has property keys

        if len(keys) > 0:
            txt = (f"You asked to use {keys} properties in Baryonification. You must pass a ParamTabulatedProfile "
                   f"pr BaryonificationClass as the model. You have passed {type(self.model)} instead. "
                   f"If you did pass in a BaryonificationClass make sure you passed in addition params using "
                   f"the other_params option.")
            assert isinstance(self.model, (ParamTabulatedProfile, BaryonificationClass)), txt
        
        for j in tqdm(range(self.HaloNDCatalog.cat.size), desc = 'Baryonifying matter', disable = not self.verbose):

            M_j = self.HaloNDCatalog.cat['M'][j]
            x_j = self.HaloNDCatalog.cat['x'][j]
            y_j = self.HaloNDCatalog.cat['y'][j]
            z_j = self.HaloNDCatalog.cat['z'][j] #THIS IS A CARTESIAN COORDINATE, NOT REDSHIFT
            o_j = {key : self.HaloNDCatalog.cat[key][j] for key in keys} #Other properties
            
            a_j = 1/(1 + self.HaloNDCatalog.redshift)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc
            R_q = self.epsilon_max * R_j/a_j #The radius for querying points, in comoving coords
            R_q = np.clip(R_q, 0, L/2) #Can't query distances more than half box-size.
            
            if is2D:
                
                inds = self.tree.query_ball_point([x_j, y_j], R_q)
                dx   = self.ParticleSnapshot.cat['x'][inds] - x_j
                dy   = self.ParticleSnapshot.cat['y'][inds] - y_j
                d    = self.compute_distance(dx, dy)

                x_hat = self.enforce_periodicity(dx)/d
                y_hat = self.enforce_periodicity(dy)/d

                #Compute the displacement needed
                offset = self.model.displacement(d, M_j, a_j, **o_j)
                offset = np.where(np.isfinite(offset), offset, 0)
                tot_offsets[inds] += np.vstack([offset*x_hat, offset*y_hat]).T
                
            
            else:
                inds = self.tree.query_ball_point([x_j, y_j, z_j], R_q)
                dx   = self.ParticleSnapshot.cat['x'][inds] - x_j
                dy   = self.ParticleSnapshot.cat['y'][inds] - y_j
                dz   = self.ParticleSnapshot.cat['z'][inds] - z_j
                d    = self.compute_distance(dx, dy, dz)

                x_hat = self.enforce_periodicity(dx)/d
                y_hat = self.enforce_periodicity(dy)/d
                z_hat = self.enforce_periodicity(dz)/d

                #Compute the displacement needed
                offset = self.model.displacement(d, M_j, a_j, **o_j)
                offset = np.where(np.isfinite(offset), offset, 0)
                tot_offsets[inds] += np.vstack([offset*x_hat, offset*y_hat, offset*z_hat]).T
                
            
        new_cat = self.ParticleSnapshot.cat.copy()
        
        new_cat['x'] += tot_offsets[:, 0]
        new_cat['y'] += tot_offsets[:, 1]
        
        if not is2D: new_cat['z'] += tot_offsets[:, 2]
            
        for i in ['x', 'y'] + ([] if self.ParticleSnapshot.is2D else ['z']):
            
            new_cat[i]  = np.where(new_cat[i] > L, new_cat[i] - L, new_cat[i])
            new_cat[i]  = np.where(new_cat[i] < 0, new_cat[i] + L, new_cat[i])

        return new_cat