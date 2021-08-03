import numpy as np

def constant_weight(*args, weight=1.):
    return weight

def rising_weight(iteration: int, *,
                  iteration_rise=4000, iteration_max=10000, max_weight=1., min_weight=0., slope=1) -> float:
    """ define weight that rises from min_weight to max_weight, from iteration_rise to iteration_max """
    assert slope % 2, "slope must be odd"
    weight = (iteration - iteration_rise)/(iteration_max - iteration_rise)
    weight = weight**slope
    weight *= max_weight - min_weight
    weight += min_weight
    weight = np.min((weight, max_weight))
    weight = np.max((weight, min_weight))
    return weight

def falling_weight(iteration: int, *,
                   iteration_fall=4000, iteration_min=10000, max_weight=1, min_weight=0., slope=1) -> float:
    """ define weight that rises from min_weight to max_weight, from iteration_fall to iteration_min """
    assert slope % 2, "slope must be odd"
    weight = (iteration_min - iteration)/(iteration_min - iteration_fall)
    weight = weight**slope
    weight *= max_weight - min_weight
    weight += min_weight
    weight = np.min((weight, max_weight))
    weight = np.max((weight, min_weight))
    return weight

def cyclical_weight(iteration: int, *,
                    full_cycle=10000, rise_cycle=5000, max_weight=1., min_weight=0., slope=1) -> float:
    """ define weight that rises for rise_cycle iterations from min_weight to max_weight,
        stays constant or rest of full_cycle and then starts again at min_weight
    """
    cycle_iteration = iteration % full_cycle
    weight = rising_weight(cycle_iteration, iteration_rise=0, iteration_max=rise_cycle, max_weight=max_weight, min_weight=min_weight, slope=slope)
    return weight

