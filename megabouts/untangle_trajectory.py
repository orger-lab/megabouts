def untangle_trajectory(data, distance=0.9):
    """Filters a trajectory signal by removing points that too close to each other in space and in time.
    Removed points are replace by a linear interpolation between two consecutive points that are far away from each other.

    Parameters
    ----------
    data : 2xN array
        The array to be filtered

    distance : float
        The maximum distance for two points to be considerer too close to each other.
    """

    pivot_pos = data[:, 0]
    pivot_idx = 0

    for i in range(1, data.shape[1]):

        if(np.linalg.norm(pivot_pos - data[:, i]) > distance):

            start_pos = pivot_pos
            stop_pos = data[:, i]
            n = i - pivot_idx + 1

            data[0, pivot_idx: i+1] = np.linspace(start_pos[0], stop_pos[0], n)
            data[1, pivot_idx: i+1] = np.linspace(start_pos[1], stop_pos[1], n)

            pivot_pos = data[:, i]
            pivot_idx = i

    return data


def untangle_trajectoryT(data, distance=0.9):
    """Filters a trajectory signal by removing points that too close to each other in space and in time.
    Removed points are replace by a linear interpolation between two consecutive points that are far away from each other.

    Parameters
    ----------
    data : Nx2 array
        The array to be filtered

    distance : float
        The maximum distance for two points to be considerer too close to each other.
    """

    pivot_pos = data[0,:]
    pivot_idx = 0

    for i in range(1, data.shape[0]):

        if(np.linalg.norm(pivot_pos - data[i,:]) > distance):

            start_pos = pivot_pos
            stop_pos = data[i,:]
            n = i - pivot_idx + 1

            data[pivot_idx: i+1,0] = np.linspace(start_pos[0], stop_pos[0], n)
            data[pivot_idx: i+1,1] = np.linspace(start_pos[1], stop_pos[1], n)

            pivot_pos = data[i,:]
            pivot_idx = i

    return data