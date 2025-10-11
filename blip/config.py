import configparser

# FIXME remove the imports below (avoid shotgun parsing i.e. acting on configs
# before we even know if they are valid)
import numpy as np
import pickle


# FIXME the `resume` parameter should not be here since it's irrelevant to
# parsing the config
def parse_config(paramsfile: str, resume: bool):
    "Parse a configuration file. Returns (params, inj, misc)."

    # TODO clearly specify what counts as valid here. A good start would be
    # to use TypedDicts.
    params = {}
    inj = {}
    misc = {}

    # TODO use mapping protocol access instead of the legacy .get() API
    # TODO set defaults (not fallbacks!) where applicable
    config = configparser.ConfigParser()
    config.read(paramsfile)

    # Params Dict
    params["fmin"] = float(config.get("params", "fmin"))
    params["fmax"] = float(config.get("params", "fmax"))
    params["dur"] = float(config.get("params", "duration"))
    params["seglen"] = float(config.get("params", "seglen", fallback=1e5))
    params["fs"] = float(config.get("params", "fs", fallback=0.25))
    params["Shfile"] = config.get("params", "Shfile", fallback="LISA_2017_PSD_M.npy")
    params["load_data"] = int(config.get("params", "load_data", fallback=0))
    params["datatype"] = str(config.get("params", "datatype", fallback="strain"))
    params["datafile"] = str(config.get("params", "datafile", fallback=None))
    ## default fref is 1mHz
    ## until we update the prior structure, using other reference frequencies may lead to unintended behavior in the prior bounds
    ## in the new prior structure, the default astrophysical prior bounds should scale with fref
    params["fref"] = float(config.get("params", "fref", fallback=0.001))
    if params["fref"] != 0.001:
        print(
            "Warning: reference frequency set to a value other than 1 mHz (fref={} Hz).".format(
                params["fref"]
            )
        )
        print(
            "This may lead to unexpected prior behavior, as the prior bounds are based on a reference frequency of 1 mHz!"
        )

    params["model"] = str(config.get("params", "model"))

    ## get the model fixed values, passed as a dict
    fixedvals = eval(str(config.get("params", "fixedvals", fallback="None")))

    params["fixedvals"] = {}
    if fixedvals is not None:
        ## enforce that the keys of the fixedvals dict correspond to the desired models
        ## at this point we will only make sure that all the keys are in the model string
        submodel_names = params["model"].split("+")
        if not np.all([key in submodel_names for key in fixedvals.keys()]):
            raise ValueError(
                "Fixedvals dictionary has an invalid key. Fixedvals must be provided as a nested dictionary with top-level keys corresponding to the models specified in the 'model' parameter."
            )
        ## build the fixedvals dict
        ## some quantities we want to use as log values, so convert them
        log_list = ["Np", "Na", "omega0", "fbreak", "fcut", "fscale"]
        for submodel_name in fixedvals.keys():
            params["fixedvals"][submodel_name] = {}
            for name in fixedvals[submodel_name].keys():
                if name in log_list:
                    new_name = "log_" + name
                    params["fixedvals"][submodel_name][new_name] = np.log10(
                        fixedvals[submodel_name][name]
                    )
                else:
                    params["fixedvals"][submodel_name][name] = fixedvals[submodel_name][
                        name
                    ]

    params["alias"] = eval(str(config.get("params", "alias", fallback="{}")))

    params["tdi_lev"] = str(config.get("params", "tdi_lev", fallback="xyz"))
    params["lisa_config"] = str(
        config.get("params", "lisa_config", fallback="orbiting")
    )
    params["nside"] = int(config.get("params", "nside"))
    params["model_basis"] = str(config.get("params", "model_basis", fallback="pixel"))
    params["tstart"] = float(config.get("params", "tstart", fallback=0))

    ## see if we need to initialize the spherical harmonic subroutines
    sph_check = [
        sublist.split("-")[0].split("_")[-1] for sublist in params["model"].split("+")
    ]

    # Injection Dict
    inj["doInj"] = int(config.get("inj", "doInj"))
    inj["loadInj"] = int(config.get("inj", "loadInj", fallback=0))
    inj["inj_only"] = int(config.get("inj", "inj_only", fallback=0))

    if inj["doInj"]:
        ## first see if we are loading the injection
        if inj["loadInj"]:
            if inj["inj_only"]:
                raise ValueError(
                    "Both loadInj and inj_only flags are set to True. This won't accomplish anything..."
                )
            inj["injdir"] = str(config.get("inj", "injdir"))
            ## get the already-generated injection dict
            with open(inj["injdir"] + "/config.pickle", "rb") as paramfile:
                ## things are loaded from the pickle file in the same order they are put in
                loaded_params = pickle.load(paramfile)
                loaded_inj = pickle.load(paramfile)
            ## for this to work, all params that impact the data/response times/frequencies/types can't change
            ## but you can change e.g., recovery model, sampler, etc.
            required_immutable = [
                "fmin",
                "fmax",
                "dur",
                "seglen",
                "fs",
                "nside",
                "tstart",
                "lisa_config",
                "tdi_lev",
                "datatype",
            ]
            requirements_violated = [
                requirement
                for requirement in required_immutable
                if params[requirement] != loaded_params[requirement]
            ]
            if len(requirements_violated) > 0:
                raise ValueError(
                    "Loaded injection is incompatible with specified configuration due to mismatches in the following config settings: {}".format(
                        requirements_violated
                    )
                )
            ## update the injection dictionary with the loaded one
            inj |= loaded_inj
            ## reset the top-level injection flags to their original state
            inj["loadInj"] = True
            inj["inj_only"] = False

        ## otherwise make a new one
        else:
            inj["injection"] = str(config.get("inj", "injection"))

            ## get the injection basis
            inj["inj_basis"] = str(config.get("inj", "inj_basis", fallback="pixel"))

            ## get the injection truevals, passed as a dict
            truevals = eval(str(config.get("inj", "truevals")))
            ## enforce that the keys of the truevals dict correspond to the injected models
            ## at this point we will only make sure that all the keys are in the injection string
            ## (not all injections will have truevals; e.g., population injections)
            inj_component_names = inj["injection"].split("+")
            if not np.all([key in inj_component_names for key in truevals.keys()]):
                raise ValueError(
                    "Truevals dictionary has an invalid key. Truevals must be provided as a nested dictionary with top-level keys corresponding to the injections specified in the 'injection' parameter."
                )

            ## injection per-component multithreading
            inj["parallel_inj"] = int(config.get("inj", "parallel_inj", fallback=0))

            ## if parallel_inj is True but there is only one component, set parallel_inj to False
            #            if len(inj_component_names) == 1 and inj['parallel_inj']:
            #                inj['parallel_inj'] = 0

            if inj["parallel_inj"]:
                inj["inj_nthread"] = int(
                    config.get("inj", "inj_nthread", fallback=len(inj_component_names))
                )
                inj["response_nthread"] = int(
                    config.get("inj", "response_nthread", fallback=1)
                )
                ## give preference to response multi-threading
                if inj["inj_nthread"] > 1 and inj["response_nthread"] > 1:
                    print(
                        "Warning: you have set both inj_nthread and response_nthread > 1."
                    )
                    print(
                        "The Multiprocessing package does not allow workers to spawn additional workers."
                    )
                    print(
                        "Giving precedence to response multiprocessing, setting inj_nthread=1."
                    )
                    inj["inj_nthread"] = 1
                elif inj["inj_nthread"] == 1 and inj["response_nthread"] == 1:
                    inj["parallel_inj"] = 0

            else:
                inj["inj_nthread"] = 1

            ## build the truevals dict
            ## some quantities we want to use as log values, so convert them
            inj["truevals"] = {}
            log_list = ["Np", "Na", "omega0", "fbreak", "fcut", "fscale"]
            for component_name in truevals.keys():
                inj["truevals"][component_name] = {}
                for name in truevals[component_name].keys():
                    if name in log_list:
                        new_name = "log_" + name
                        inj["truevals"][component_name][new_name] = np.log10(
                            truevals[component_name][name]
                        )
                    else:
                        inj["truevals"][component_name][name] = truevals[
                            component_name
                        ][name]

        ## add injections to the spherical harmonic check if needed
        sph_check = sph_check + [
            sublist.split("-")[0].split("_")[-1]
            for sublist in inj["injection"].split("+")
        ]

    ## pop out to set sph flags
    params["sph_flag"] = "sph" in sph_check  # or ('hierarchical' in sph_check)
    ## set sph flag to false if both inj and model basis are pixel
    if params["model_basis"] == "sph" or inj.get("inj_basis", "sph") == "sph":
        params["sph_flag"] = True
        params["lmax"] = int(config.get("params", "lmax"))

    ## some final flag, injection parameter setting if we aren't loading the Injection directly
    if inj["doInj"] and not inj["loadInj"]:

        ## similarly, set inj sph flag to False if we're doing pixel basis injections
        if inj["inj_basis"] == "pixel" and not ("sph" in sph_check):
            inj["sph_flag"] = False
        ## but if we're also explicitly doing a sph injection, set it to true
        elif inj["inj_basis"] == "pixel" and ("sph" in sph_check):
            inj["sph_flag"] = True
        else:
            inj["sph_flag"] = np.any(
                [(item not in ["noise", "isgwb"]) for item in sph_check]
            )
        ## set pop flag if spatial and/or spectral injection is a population
        inj["pop_flag"] = ("population" in sph_check) or (
            "population"
            in [
                sublist.split("-")[0].split("_")[0]
                for sublist in inj["injection"].split("+")
            ]
        )

        if inj["sph_flag"]:
            try:
                inj["inj_lmax"] = int(config.get("inj", "inj_lmax"))
            except configparser.NoOptionError as err:
                if params["sph_flag"]:
                    print(
                        "Performing a spherical harmonic basis injection and inj_lmax has not been specified. Injection and recovery will use same lmax (lmax={}).".format(
                            params["lmax"]
                        )
                    )
                    inj["inj_lmax"] = params["lmax"]
                else:
                    print(
                        "You are trying to do a spherical harmonic injection, but have not specified lmax."
                    )
                    if "lmax" in params.keys():
                        print(
                            "Warning: using analysis lmax parameter for inj_lmax, but you are not performing a spherical harmonic analysis."
                        )
                        inj["inj_lmax"] = params["lmax"]
                    else:
                        raise err

        ## NB -- will have to change this structure to allow pop-based recovery models. But it's a good start.
        if inj["doInj"] and inj["pop_flag"]:
            inj["popdict"] = eval(
                str(config.get("inj", "population_params", fallback="None"))
            )
            ## make sure every injection population component has a corresponding entry
            inj_pop_component_names = [
                cmn for cmn in inj_component_names if "population" in cmn
            ]
            for cmn in inj_pop_component_names:
                if cmn not in inj["popdict"].keys():
                    raise KeyError(
                        "Population injection '{}' does not have a corresponding entry in the population_params dict.".format(
                            cmn
                        )
                    )
            ## step through the (possibly multiple) populations and tweak formatting for the delimiters
            ## also enforce the required keys and substitute defaults if optional setting isn't given
            required_keys = ["popfile", "columns", "delimiter"]
            pop_defaults = {"snr_cut": 7, "name": None, "coldict": None}
            for key in inj["popdict"].keys():
                ## make sure it corresponds to an injection
                if key not in inj_component_names:
                    raise KeyError(
                        "Population '{}' not in injection. Top-level keys for the population_params dict must correspond to an injection.".format(
                            key
                        )
                    )
                ## enforce required keys
                for rk in required_keys:
                    if rk not in inj["popdict"][key].keys():
                        raise KeyError(
                            "population_params dict missing required key: '{}'".format(
                                rk
                            )
                        )
                ## set defaults
                for dk in pop_defaults.keys():
                    if dk not in inj["popdict"][key].keys():
                        print(
                            "No value found for populations parameter '{}' in population_params dict for injection component '{}'. Setting to default ({}).".format(
                                dk, key, pop_defaults[dk]
                            )
                        )
                        inj["popdict"][key][dk] = pop_defaults[dk]
                ## formatting
                if inj["popdict"][key]["delimiter"] == "space":
                    inj["popdict"][key]["delimiter"] = " "
                elif inj["popdict"][key]["delimiter"] == "tab":
                    inj["popdict"][key]["delimiter"] == "\t"

    # some run parameters
    params["out_dir"] = str(config.get("run_params", "out_dir"))

    params["doPreProc"] = int(config.get("run_params", "doPreProc", fallback=0))
    params["input_spectrum"] = str(
        config.get("run_params", "input_spectrum", fallback="data_spectrum.npz")
    )
    params["projection"] = str(config.get("run_params", "projection", fallback="E"))
    params["FixSeed"] = int(config.get("run_params", "FixSeed", fallback=0))
    if params["FixSeed"]:
        params["seed"] = int(config.get("run_params", "seed"))
    misc["nthread"] = int(config.get("run_params", "Nthreads", fallback=1))
    misc["N_GPU"] = int(config.get("run_params", "N_GPU", fallback=0))

    params["colormap"] = str(config.get("run_params", "colormap", fallback="magma"))

    ## sampler selection
    params["sampler"] = str(config.get("run_params", "sampler"))

    ## only numpyro has GPU support
    if misc["N_GPU"] > 0 and params["sampler"] not in ["numpyro", "numpyro_nested"]:
        raise ValueError(
            "Only numpyro supports GPU acceleration but N_GPU ({}) > 0 and sampler is {}.".format(
                misc["N_GPU"], params["sampler"]
            )
        )

    ## sampler setup and late-time imports to reduce dependencies
    ## dynesty
    if params["sampler"] == "dynesty":
        misc["nlive"] = int(config.get("run_params", "nlive", fallback=800))
        params["sample_method"] = str(
            config.get("run_params", "sample_method", fallback="rwalk")
        )
    ## emcee
    elif params["sampler"] == "emcee":
        params["Nburn"] = int(config.get("run_params", "Nburn", fallback=1000))
        params["Nsamples"] = int(config.get("run_params", "Nsamples", fallback=1000))
    ## numpyro
    elif params["sampler"] == "numpyro":
        params["show_progress"] = int(
            config.get("run_params", "show_progress", fallback=1)
        )
        params["Nburn"] = int(config.get("run_params", "Nburn", fallback=1000))
        params["Nsamples"] = int(config.get("run_params", "Nsamples", fallback=1000))

    # TODO either remove this branch entirely or uncomment it.
    # NOTE numpyro_nested_engine has been commented out of blip.src.numpyro_engine
    # so this code path wouldn't work if it was kept uncommented.
    #
    # ## numpyro nested sampling
    # elif params['sampler'] == 'numpyro_nested':
    #     if nthread > 1:
    #         os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count={}'.format(nthread)
    #     from blip.src.numpyro_engine import numpyro_nested_engine
    #     params['show_progress'] = int(config.get("run_params", "show_progress", fallback=1))
    #     params['Nburn'] = int(config.get("run_params", "Nburn",fallback=1000))
    #     params['Nsamples'] = int(config.get("run_params", "Nsamples",fallback=1000))

    else:
        raise ValueError(
            "Unknown sampler. Supported samplers: 'dynesty', 'emcee', and 'numpyro'."
        )
    # checkpointing (dynesty+numpyro only for now)
    if params["sampler"] == "dynesty" or params["sampler"] == "numpyro":
        params["checkpoint"] = int(config.get("run_params", "checkpoint", fallback=0))
        ## numpyro's checkpoint_interval is in number of samples, vs. seconds for dynesty
        if params["sampler"] == "numpyro":
            params["checkpoint_at"] = str(
                config.get("run_params", "checkpoint_at", fallback="end")
            )
            if params["checkpoint_at"] == "interval":
                params["checkpoint_interval"] = int(
                    config.get("run_params", "checkpoint_interval", fallback=100)
                )
            params["additional_samples"] = int(
                config.get("run_params", "additional_samples", fallback=0)
            )
            if params["additional_samples"] == 0:
                params["additional_samples"] = None
        else:
            params["checkpoint_interval"] = int(
                config.get("run_params", "checkpoint_interval", fallback=3600)
            )

    # Fix random seed
    if params["FixSeed"]:
        from blip.tools.SetRandomState import SetRandomState as setrs

        seed = params["seed"]
        misc["randst"] = setrs(seed)
    else:
        if params["checkpoint"]:
            raise TypeError(
                "Checkpointing without a fixed seed is not supported. Set 'FixSeed' to true and specify 'seed'."
            )
        if resume:
            raise TypeError(
                "Resuming from a checkpoint requires re-generation of data, so the random seed MUST be fixed."
            )
        misc["randst"] = None

    return params, inj, misc
