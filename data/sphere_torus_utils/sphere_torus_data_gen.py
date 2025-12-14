import numpy as np


def computeTrueNormal(phi, theta, mode):
    """
    Compute the true normal vectors for each point in the dataset.

    Parameters:
    - phi: Array of random angles phi for each point
    - theta: Array of random angles theta for each point
    - mode: Mode of the dataset generation ("sphere" or "torus")

    Returns:
    - tangents: Array of tangent vectors for each point
    - normals: Array of normal vectors for each point
    """
    N = len(phi)
    tangents = np.zeros((N, 3, 2))
    if mode == "sphere":
        # Compute tangent vectors for the sphere mode
        tangents[:, 0, 0] = -np.sin(phi)
        tangents[:, 1, 0] = np.cos(phi)
        tangents[:, 2, 0] = 0

        tangents[:, 0, 1] = np.cos(phi) * np.cos(theta)
        tangents[:, 1, 1] = np.sin(phi) * np.cos(theta)
        tangents[:, 2, 1] = -np.sin(theta)

        # Compute normal vectors as cross product of tangent vectors
        normals = np.cross(
            tangents[:, :, 0],
            tangents[:, :, 1],
        )
    elif mode == "torus":
        # Compute tangent vectors for the torus mode
        tangents[:, 0, 0] = -np.sin(phi) * np.cos(theta)
        tangents[:, 1, 0] = np.cos(phi) * np.sin(theta)
        tangents[:, 2, 0] = np.cos(phi)

        tangents[:, 0, 1] = -np.sin(theta)
        tangents[:, 1, 1] = np.cos(theta)
        tangents[:, 2, 1] = 0

        # Compute normal vectors as cross product of tangent vectors
        normals = np.cross(
            tangents[:, :, 0],
            tangents[:, :, 1],
        )
    else:
        raise ValueError("Invalid mode")
    return tangents, normals


def circle(theta, phi, R):
    """
    Generate the sphere given spherical coordinates

    Parameters:
    - theta: Array of random angles theta for each point
    - phi: Array of random angles phi for each point
    - R: radius of the sphere

    Returns:
    - data: Array of data points
    """
    X = R * np.cos(phi) * np.sin(theta)
    Y = R * np.sin(phi) * np.sin(theta)
    Z = R * np.cos(theta)
    data = np.stack((X, Y, Z), axis=1)
    return data


def inverse_circle(data):
    """
    Generate the spherical coordinates given the sphere

    Parameters:
    - data: Array of data points

    Returns:
    - theta: Array of random angles theta for each point
    - phi: Array of random angles phi for each point
    """
    X = data[:, 0]
    Y = data[:, 1]
    Z = data[:, 2]
    phi = np.arctan2(Y, X)
    theta = np.arccos(Z)
    phi[phi < 0] += 2 * np.pi
    return theta, phi


def genData(N, R, r, mode, distort=False):
    """
    Generate random data points on a sphere or torus.

    Parameters:
    - N: Number of points to generate
    - R: Major radius of the torus (for torus mode)
    - r: Minor radius of the torus (for torus mode)
    - mode: Mode of the dataset generation ("sphere" or "torus")
    - distort: Boolean indicating whether to distort the data

    Returns:
    - data: Array of data points
    - theta: Array of random angles theta for each point
    - phi: Array of random angles phi for each point
    """
    # Generate random angles for each point
    theta = np.random.uniform(0, 2 * np.pi, N)
    phi = np.random.uniform(0, 2 * np.pi, N)

    if mode == "torus":
        if distort:
            print("warning theta and phi are distorted")
            theta = np.power(theta / np.pi - 1, 3) * np.pi + np.pi
            phi = np.power(phi / np.pi - 1, 3) * np.pi + np.pi
        # Calculate the x, y, and z coordinates of each point for the torus mode
        X = (R + r * np.cos(phi)) * np.cos(theta)
        Y = (R + r * np.cos(phi)) * np.sin(theta)
        Z = r * np.sin(phi)
        data = np.stack((X, Y, Z), axis=1)

    elif mode == "sphere":
        data = np.random.normal(size=(N, 3))
        data = data / np.linalg.norm(data, axis=1)[:, None]
        theta, phi = inverse_circle(data)
        if distort:
            print("warning theta and phi are distorted")
            theta = (
                (
                    np.power(
                        2 * theta / np.pi - 1,
                        3,
                    )
                    + 1
                )
                * np.pi
                / 2
            )
            phi = (np.power(phi / np.pi - 1, 3) + 1) * np.pi
            # theta = np.power(theta/np.pi-1, 3)*np.pi+np.pi
            # phi = np.power(phi/np.pi-1, 3)*np.pi+np.pi
        data = circle(theta, phi, R)
    else:
        raise ValueError("Invalid mode")
    return data, theta, phi


def genSphere(N, tangentDim, ambientDim):
    """
    Generate random data points on a tangentDim-sphere
    in an ambientDim-dimensional space.

    Parameters:
    - N: Number of points to generate
    - tangentDim: Dimension of the tangent space
    - ambientDim: Dimension of the ambient space

    Returns:
    - data: Array of data points
    """
    data = np.random.normal(size=(N, ambientDim))
    codimension = ambientDim - tangentDim
    for i in range(codimension - 1):
        data[:, i - 1] = 0
    data = data / np.linalg.norm(data, axis=1).reshape(-1, 1)  # Normalize data points
    return data  # Return data points


def pathsToCloud(pathData, d) -> np.ndarray:
    """
    This function reshapes the data from the paths into a cloud of points
    going from shape (pathCount, N, 3) to (pathCount*N, 3)

    Parameters:
        - pathData: the data from the paths
        - d: the dimension of the data

    Returns:
        - the reshaped data
    """

    if isinstance(pathData, np.ndarray) and len(pathData.shape) == 2:
        assert pathData.shape[-1] == d
        return pathData

    totalCount = 0
    for path in pathData:
        totalCount += path.shape[0]
        assert path.shape[-1] == d

    cloudData = np.zeros((totalCount, d))
    count = 0
    for path in pathData:
        amountToAdd = path.shape[0]
        cloudData[count : count + amountToAdd] = path
        count += amountToAdd

    return cloudData


def angularMomentumPrime(I1, I2, I3, x):
    """
    This function returns the derivative of the angular momentum
    given the current state of the system. The specific system we're
    modelling here is given by the following diff eq:
    y1' = (I3 - I2) * y2 * y3
    y2' = (I1 - I3) * y3 * y1
    y3' = (I2 - I1) * y1 * y2

    Physics convention are to replace I1, I2, I3 with their reciprocal.

    Parameters:
        - I1: first principal moment of inertia
        - I2: second principal moment of inertia
        - I3: third principal moment of inertia
        - pathData: the data from the paths in shape (pathCount, N, 3)

    Returns:
        - pathDataPrime
    """
    pathDataPrime = np.zeros(x.shape)
    y1, y2, y3 = x[..., 0], x[..., 1], x[..., 2]
    pathDataPrime[..., 0] = (I3 - I2) * y2 * y3
    pathDataPrime[..., 1] = (I1 - I3) * y3 * y1
    pathDataPrime[..., 2] = (I2 - I1) * y1 * y2
    return pathDataPrime
