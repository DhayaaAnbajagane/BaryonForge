import joblib
import numpy as np

__all__ = ['SimpleParallel', 'SplitJoinParallel']

from ..Runners import BaryonifyShell, BaryonifyGrid, BaryonifySnapshot, PaintProfilesAnisShell, DefaultRunner

class SimpleParallel(object):
    """
    A class to execute a list of Runner objects in parallel using a joblib instance.

    The `SimpleParallel` class allows for the parallel execution of tasks encapsulated by Runner objects.
    It utilizes joblib's parallel processing capabilities to distribute the workload across multiple CPU cores.
    This class is particularly useful for running computationally intensive tasks concurrently, thereby reducing
    the overall processing time.

    Parameters
    ----------
    Runner_list : list
        A list of Runner objects, each of which must have a `process()` method that defines the task to be executed.
    
    njobs : int, optional
        The number of jobs (processes) to run in parallel. If set to -1, the number of jobs is set to the number of CPUs 
        available. Default is -1.


    Methods
    -------
    single_run(i, Runner)
        Executes the `process()` method of a single Runner and returns its index and output.
    
    process()
        Executes all Runners in the Runner_list in parallel, returning their outputs in the original order.

    Examples
    --------
    >>> runners = [Runner1(), Runner2(), Runner3()]
    >>> parallel_executor = SimpleParallel(runners, njobs=2)
    >>> results = parallel_executor.process()

    Notes
    -----
    - The `Runner` objects must have a `process()` method implemented. This method will be called during the parallel
      execution.
    - The number of parallel jobs will be adjusted to match the number of Runners if there are fewer Runners than
      available CPUs.
    - Pickling errors can occur if the cosmology object (which is stored as an attribute in many classes) contains
      SwigPyObjects that are not pickleable. This happens with P(k) calculations. The baryonification pipeline
      natively destroys these problem objects. If you are facing issues, perform an explicit check of this.

    See Also
    --------
    joblib.Parallel : The underlying parallel execution library used for running the tasks.
    joblib.delayed : A function to wrap the tasks to be executed in parallel.

    """
    
    def __init__(self, Runner_list, njobs = -1):
        
        self.Runner_list = Runner_list
        self.njobs = njobs if njobs != -1 else joblib.externals.loky.cpu_count()

        if len(Runner_list) < self.njobs:
            self.njobs = len(Runner_list)
            print(f"You asked for more processors than needed. Setting n_jobs = {self.njobs}")
    
    
    def single_run(self, i, Runner):
        """
        Executes the `process()` method of a single Runner and returns its index and output.

        This method is used to run a single Runner object. It returns the index of the Runner and its output,
        which allows for the ordered aggregation of results.

        Parameters
        ----------
        i : int
            The index of the Runner in the Runner_list.
        
        Runner : object
            An instance of a Runner object, which must have a `process()` method.

        Returns
        -------
        tuple
            A tuple containing the index of the Runner and the output of its `process()` method.
        """
        
        return i, Runner.process()
    
    
    def process(self):
        """
        Executes all Runners in the Runner_list in parallel, returning their outputs in the original order.

        This method uses joblib's Parallel and delayed functions to run each Runner's `process()` method in parallel.
        The outputs are collected and sorted based on the original order of the Runner_list.

        Returns
        -------
        list
            A list of outputs from the `process()` methods of each Runner, in the order they were provided in Runner_list.
        """
        
        with joblib.parallel_backend("loky"):
            jobs    = [joblib.delayed(self.single_run)(i, Runner) for i, Runner in enumerate(self.Runner_list)]
            outputs = joblib.Parallel(n_jobs = self.njobs, verbose=10)(jobs)

        #Sort them so they are in the same order as they were input
        ordered_outputs = [0] * len(outputs)
        for o in outputs: ordered_outputs[o[0]] = o[1]
            
        return ordered_outputs
    
    
class SplitJoinParallel(object):
    """
    A class to split a single Runner task into multiple parallel tasks and join the results.

    The `SplitJoinParallel` class takes a single Runner object and splits its task into multiple smaller tasks
    to be run in parallel. It uses joblib for parallel execution, and the results are combined (joined) after
    all parallel tasks are completed. This approach is particularly useful for handling large datasets or 
    computationally intensive tasks by distributing the workload across multiple CPU cores.

    This class is currently not designed to work with the Baryonification operation and will raise an 
    error if it is passed any Baryonification runner.
    
    Parameters
    ----------
    Runner : object
        A Runner object that defines the task to be executed. This object must have methods and attributes
        necessary for splitting the task, such as `HaloLightConeCatalog`, `LightconeShell`, `cosmo`, etc.
    
    seed : int
        A seed value for initializing the random number generator to ensure reproducibility when shuffling
        the halo catalog (shufflying is done to load balance the jobs). Default is 42.

    njobs : int, optional
        The number of jobs (processes) to run in parallel. If set to -1, the number of jobs is set to the number
        of CPUs available. Default is -1.

    Attributes
    ----------
    Runner : object
        The original Runner object that is being split for parallel processing.
    
    seed : int
        A seed value used when shuffling the halo catalog. Default is 42.
    
    njobs : int
        The number of jobs (processes) that will be run in parallel. This is adjusted based on the number of 
        available CPUs and the length of the Runner_list.
    
    Runner_list : list
        A list of Runner objects, each representing a subset of the original task to be run in parallel.

    Methods
    -------
    split_run(Runner)
        Splits the original Runner task into multiple smaller Runner tasks for parallel processing.
    
    single_run(Runner)
        Executes the `process()` method of a single Runner and returns its output.
    
    process()
        Executes all Runners in the Runner_list in parallel, returning the combined output.

    Examples
    --------
    >>> runner = SomeRunnerClass()
    >>> split_join_executor = SplitJoinParallel(runner, njobs=4)
    >>> result = split_join_executor.process()

    Notes
    -----
    - The `Runner` object must have specific attributes like `HaloLightConeCatalog`, `LightconeShell`, `cosmo`,
      `model`, `mass_def`, `epsilon_max`, and `use_ellipticity`.
    - The number of parallel jobs will be adjusted to match the number of splits if there are fewer splits than
      available CPUs.
    - Pickling errors can occur if the cosmology object (which is stored as an attribute in many classes) contains
      SwigPyObjects that are not pickleable. This happens with P(k) calculations. The baryonification pipeline
      natively destroys these problem objects. If you are facing issues, perform an explicit check of this.

    See Also
    --------
    joblib.Parallel : The underlying parallel execution library used for running the tasks.
    joblib.delayed : A function to wrap the tasks to be executed in parallel.

    """

    def __init__(self, Runner, njobs = -1, seed = 42):
        """
        Initializes the SplitJoinParallel class with a Runner and the number of jobs.

        Parameters
        ----------
        Runner : object
            A Runner object that defines the task to be executed. This object must have methods and attributes
            necessary for splitting the task, such as `HaloLightConeCatalog`, `LightconeShell`, `cosmo`, etc.
        
        njobs : int, optional
            The number of jobs (processes) to run in parallel. If set to -1, the number of jobs is set to the number
            of CPUs available. Default is -1.
        """
        
        #The SplitJoin runner only works when the final output can be linearly summed across different batches of halos.
        #A painting operation is an ideal use case, while Baryonification is not, so do an explicit check here.
        #The anisotropic painter normalizes by the total mass map of *all* halos, so it cannot be split either.
        #The splitting also relies on the HEALPix runner interface (HaloLightConeCatalog, LightconeShell).
        text = f"Runner of type {type(Runner)} is not supported for SplitJoinParallel."
        assert not isinstance(Runner, (BaryonifyGrid, BaryonifyShell, BaryonifySnapshot, PaintProfilesAnisShell)), text
        assert isinstance(Runner, DefaultRunner), text + " Only HEALPix (lightcone shell) painting runners are supported."

        self.Runner = Runner
        self.seed   = seed
        self.njobs  = njobs if njobs != -1 else joblib.externals.loky.cpu_count()
        
        self.Runner_list = self.split_run(self.Runner)
        
        
    def split_run(self, Runner):
        """
        Splits the original Runner task into multiple smaller Runner tasks for parallel processing.

        This method divides the halo catalog from the Runner into multiple subsets. Each subset is used to
        create a new Runner, which will process only that subset. The splitting process includes shuffling
        the catalog to optimize load balancing across processes.

        Parameters
        ----------
        Runner : object
            The original Runner object to be split.

        Returns
        -------
        Runner_list : list
            A list of new Runner objects, each configured to process a subset of the original Runner's task.
        """
        
        HaloCat  = Runner.HaloLightConeCatalog
        Shell    = Runner.LightconeShell
        cosmo    = Runner.cosmo
        model    = Runner.model
        mass_def = Runner.mass_def
        eps_max  = Runner.epsilon_max
        ellip    = Runner.use_ellipticity
        pixsize  = Runner.include_pixel_size
        
        #Now split
        
        catalog   = HaloCat.cat
        Nsplits   = self.njobs
        Ntotal    = len(catalog)
        Npersplit = int(np.ceil(Ntotal/Nsplits))

        #Randomize catalog ordering. This helps optimize the parallelization. Else if 
        #low redshift halos are all the start, then handful of processes will be overburdened 
        #while the rest are just sitting idle.
        HaloCat     = HaloCat[np.random.default_rng(self.seed).choice(Ntotal, size = Ntotal, replace = False)] 
        
        empty_shell = type(Shell)(map = np.zeros_like(Shell.map), cosmo = cosmo, redshift = Shell.redshift)
        
        Runner_list = []
        for i in range(Nsplits):
            
            start = i*Npersplit
            end   = (i + 1)*Npersplit
            
            #Halocatalog object is sliceable like a regular numpy array so can easily
            #split the halos up but keep the same data structure
            New_HaloCatalog = HaloCat[start:end]
            
            #Create a new Runner for just a subset of catalog. Has same model, map size etc.
            #Force verbose to be off as we don't want outputs for each subrun of parallel process.
            New_Runner = type(Runner)(New_HaloCatalog, empty_shell, eps_max, model, ellip, mass_def,
                                      include_pixel_size = pixsize, verbose = False)
            
            Runner_list.append(New_Runner)
        
        return Runner_list
        
    def single_run(self, Runner): 
        """
        Executes the `process()` method of a single Runner and returns its output.

        This method is used to run a single Runner object.

        Parameters
        ----------
        Runner : object
            An instance of a Runner object, which must have a `process()` method.

        Returns
        -------
        output
            The output of the Runner's `process()` method.
        """       
        
        return Runner.process()
    
    
    def process(self):
        """
        Executes all Runners in the Runner_list in parallel, returning the combined output.

        This method uses joblib's Parallel and delayed functions to run each Runner's `process()` method in parallel.
        The outputs are combined by summing them, which is appropriate if the contributions from each runner can
        be linearly summed. This is ideal for HEALPix profile painting tasks, such as `PaintProfilesShell`.
        Grid runners (eg. `PaintProfilesGrid`) and `PaintProfilesAnisShell` are not supported.

        Returns
        -------
        map_out : ndarray
            A combined map output generated by summing the outputs from each Runner's `process()` method.
        """
        
        with joblib.parallel_backend("loky"):
            jobs    = [joblib.delayed(self.single_run)(Runner) for Runner in self.Runner_list]
            outputs = joblib.Parallel(n_jobs = self.njobs, verbose=10)(jobs)

        #Sum the contributions from invidual runs. 
        #Contributions from each halo can be linearly added so this is fine
        map_out = np.sum(outputs, axis = 0)
        
        return map_out
    
    