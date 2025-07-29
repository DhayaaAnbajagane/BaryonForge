import pyccl as ccl
import numpy as np

__all__ = ['BaseGenericConcentration', 
           'GenericConcentrationDuffy08', 'GenericConcentrationKlypin11', 'GenericConcentrationPrada12',
           'GenericConcentrationDiemer15', 'GenericConcentrationIshiyama21', 'GenericConcentrationBhattacharya13']

class BaseGenericConcentration(ccl.halos.halo_model_base.Concentration):
    
    """
    Generic concentration re-mapper between halo mass definitions.

    This base class converts a concentration–mass relation defined for an input
    mass definition (``mdef_in``) to a *target* mass definition (``mass_def``),
    preserving the scale radius :math:`r_s`. Subclasses should set the class
    attributes ``cmodel`` (a concentration model factory) and ``mdef_in``
    (the input :class:`pyccl.halos.massdef.HaloMassDef`), then initialize this
    class with the desired target ``mass_def``.

    Parameters
    ----------
    mass_def : pyccl.halos.massdef.HaloMassDef
        Target halo mass definition for which concentrations will be returned
        (e.g., ``MassDef200m()``, ``MassDef200c()``, ``MassDefVir()``, etc.).

    Attributes
    ----------
    cmodel : callable
        A *factory* for a concentration model that accepts ``mass_def=`` and
        returns a callable ``c(M, a)``. It will be invoked as
        ``cmodel(mass_def=self.mdef_in)(cosmo, M, a)``.
    mdef_in : pyccl.halos.massdef.HaloMassDef
        Input mass definition on which the underlying concentration–mass model
        is calibrated.
    M_in_lo : float, optional
        Lower bound of the internal sampling range for mass translation
        (default: ``1e10``).
    M_in_hi : float, optional
        Upper bound of the internal sampling range for mass translation
        (default: ``1e16``).
    M_in_N : int, optional
        Number of logarithmically spaced samples between ``M_in_lo`` and
        ``M_in_hi`` used to construct the translation grid (default: ``100``).

    Call Signature
    --------------
    __call__(cosmo, M, a)

        Remap the concentration–mass relation to the target ``mass_def`` and
        evaluate it at the requested masses ``M``.

    Parameters
    ----------
    cosmo : pyccl.Cosmology
        Cosmology object used by pyCCL.
    M : array_like
        Halo masses *in the target mass definition* ``mass_def``. Must be
        broadcastable to a 1D array. Units must be consistent with your
        pyCCL configuration (typically :math:`M_\odot/h`).
    a : float
        Scale factor :math:`a = 1/(1+z)`.

    Returns
    -------
    c_use : numpy.ndarray
        Concentrations evaluated at ``M`` for the target mass definition
        ``mass_def``; same shape as ``M``.

    Notes
    -----
    
    The algorithm computes the ``cdelta``-``Mdelta`` relation for the input mass definition.
    Then we convert ``Mdelta`` to the target mass ``Mout`` with the right mass definition.
    We can then take the scale radius ``rs`` computed from the input mass definition and
    use it with the radius ``Rout`` of the output mass definition, to get concentration ``cout``.
    
    Examples
    --------
    Define a subclass that remaps a calibrated model from 200c to 200m:

    >>> from pyccl.halos import massdef
    >>> class MyDuffyRemapper(BaseGenericConcentration):
    ...     cmodel  = Duffy08Concentration   # factory taking mass_def=...
    ...     mdef_in = massdef.MassDef200c()
    ...
    >>> cm = MyDuffyRemapper(mass_def=massdef.MassDef200m())
    >>> c = cm(cosmo, M=[1e12, 1e13], a=1.0)
    """

    cmodel  = None
    mdef_in = None
    M_in_lo = 1e10
    M_in_hi = 1e16
    M_in_N  = 100
    name    = 'BaseGeneric'
    
    def _concentration(self, cosmo, M, a):
        """
        Evaluate the concentration–mass relation at ``M`` for the **target**
        mass definition (``self.mass_def``) by remapping from the input definition
        (``self.mdef_in``) while preserving the scale radius :math:`r_s`.

        This method builds an internal translation grid, converts masses from
        ``mdef_in`` to ``mass_def`` using :func:`pyccl.halos.mass_translator`,
        computes the implied concentrations in the target definition, and then
        interpolates to the requested masses.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            Cosmology object consumed by CCL.
        M : array_like of float
            Halo masses (in :math:`M_\odot/h`) expressed in the **target** mass definition
            ``self.mass_def``. Must satisfy
            ``self.M_in_lo < M.min()`` and ``M.max() < self.M_in_hi``.
        a : float
            Scale factor, :math:`a = 1/(1+z)`.

        Returns
        -------
        numpy.ndarray
            Concentrations ``c(M, a)`` for the target mass definition;
            same shape as ``M``.

        Raises
        ------
        AssertionError
            If any requested mass lies outside the open interval
            ``(self.M_in_lo, self.M_in_hi)``. Widen ``M_in_lo``/``M_in_hi`` or
            adjust your query if you need a broader range.
        """
        
        assert np.min(M) > self.M_in_lo, f"M_in_lo ({self.M_in_lo}) > min[M_input] ({np.min(M)})"
        assert np.max(M) < self.M_in_hi, f"M_in_hi ({self.M_in_hi}) < max[M_input] ({np.max(M)})"

        Min   = np.geomspace(self.M_in_lo, self.M_in_hi, self.M_in_N)
        cin   = self.cmodel(mass_def = self.mdef_in)(cosmo, Min, a)
        Rin   = self.mdef_in.get_radius(cosmo, Min, a)/a
        r_s   = Rin / cin  
        
        calc  = ccl.halos.mass_translator(mass_in = self.mdef_in, mass_out = self.mass_def, 
                                          concentration = self.cmodel(mass_def = self.mdef_in))
        Mout  = calc(cosmo, Min, a)
        Rout  = self.mass_def.get_radius(cosmo, Mout, a)/a
        cout  = Rout / r_s
        
        c_use = np.exp(np.interp(np.log(M), np.log(Mout), np.log(cout)))
        
        return c_use
    
    #We don't want the check to process. This should always pass
    #Because our class works for all possible mass definitions
    def _check_mass_def_strict(self, mass_def): return False
    

class GenericConcentrationDuffy08(BaseGenericConcentration):
    
    cmodel  = ccl.halos.concentration.ConcentrationDuffy08
    mdef_in = ccl.halos.massdef.MassDef200c


class GenericConcentrationKlypin11(BaseGenericConcentration):
    
    cmodel  = ccl.halos.concentration.ConcentrationKlypin11
    mdef_in = ccl.halos.massdef.MassDefVir


class GenericConcentrationPrada12(BaseGenericConcentration):
    
    cmodel  = ccl.halos.concentration.ConcentrationPrada12
    mdef_in = ccl.halos.massdef.MassDef200c


class GenericConcentrationDiemer15(BaseGenericConcentration):
    
    cmodel  = ccl.halos.concentration.ConcentrationDiemer15
    mdef_in = ccl.halos.massdef.MassDef200c


class GenericConcentrationBhattacharya13(BaseGenericConcentration):
    
    cmodel  = ccl.halos.concentration.ConcentrationBhattacharya13
    mdef_in = ccl.halos.massdef.MassDefVir


class GenericConcentrationIshiyama21(BaseGenericConcentration):
    
    cmodel  = ccl.halos.concentration.ConcentrationIshiyama21
    mdef_in = ccl.halos.massdef.MassDef200c