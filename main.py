from retention_curves import RetentionCurves
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
import numpy as np
import math
import time
import cv2

sns.set_theme()

# Define constants
REALTIME = 10  # simulation time [s]
dL = 0.25 * 0.01  # size of the block [m]
dx_PAR = dL / 0.01  # discretization parameter [-]  ratio of dL and 0.01m
S0 = 0.01  # initial saturation [-]

A = 0.31  # width of the medium [m]  for a=dL 1D semi-continuum model is used
B = 0.50  # depth of the medium [m]
M = math.floor(A / dL)  # number of blocks in a row
N = math.floor(B / dL)  # number of blocks in a column

G = 9.81  # acceleration due to gravity
THETA = 0.35  # porosity
KAPPA = 2.293577981651376e-10  # intrinsic permeability
MU = 9e-4  # dynamic viscosity
RHO = 1000  # density of water

Q0 = 8e-5  # flux at the top of boundary [m/s]
FLUX_FULL = False  # set True for flux at whole top boundary, otherwise set false
FLUX_MIDDLE = True  # set True for flux at middle of top boundary (at 1 cm), otherwise set False

# The flux from the bottom boundary is set to zero if the saturation of the respective block does not exceed a residual
# saturation. Otherwise, equation (7) is used from the paper: https://doi.org/10.1038/s41598-021-82317-x Set residual
# saturation above 1.00 in the case of zero bottom boundary flux
# SATURATION_RESIDUAL = 0.05  # residual saturation
SATURATION_RESIDUAL = 1.05  # residual saturation

# hysteresis: the gradient of transition between the main branches of the retention curve hysteresis
Kps = 1e5

# Set True if you want to use van Genuchten retention curve. = recommended
# Set False if you want to use logistic retention curve. Used in: https://doi.org/10.1038/s41598-019-44831-x
# Retention curves are defined in retention_curves.py
GENUCHTEN = True

# Definition of the initial pressure. You can either start on the main wetting branch or main draining branch
WHICH_BRANCH = "wet"

# van Genuchten parameters for 20/30 sand
ALFA_W = 0.177
N_W = 6.23
ALFA_D = 0.0744
N_D = 8.47
M_Q = 1 - 1 / N_W

# Definition of the relative permeability
LAMBDA = 0.8

TIME_INTERVAL = 1.0  # define interval in [s]
LIM_VALUE = 0.999  # instead of unity, the value very close to unity is used


def relative_permeability(saturation):
    # return saturation ** 3.5
    return saturation ** LAMBDA * (1 - (1 - saturation ** (1 / M_Q)) ** M_Q) ** 2


# Definition of the time step
dtBase = 1e-3 * 0.25  # time step [s] for dL=0.01[m]
dt = (dx_PAR ** 2) * dtBase  # time step [s], typical choice of time step parameter for parabolic equation
RATIO = 1. / dx_PAR ** 2  # dtBase/dt=1./dx_par^2
SM = dt / (THETA * dL)  # [s/m] parameter

iteration = round(REALTIME / dt)  # number of iteratioN, t=dt*iter is REALTIME

PLOT_TIME = True  # Set True for time plot of porous media flow
SAVE_DATA = True  # Set True if you want to save Saturation and Pressure data

# Set true if you want to plot the basic retention curve and its linearmodification which corresponds to the size of
# the block used for simulation
PLOT_RETENTION_CURVE = True

# DEFINITION OF THE REFERENCE BLOCK SIZE
# The crucial idea of the semi-continuum model is the scaling of the retention curve.
# For more details, see: https://doi.org/10.1038/s41598-022-11437-9

# Parameters A_WB, A_DB define the linear multiplication of the retention curve for the wetting and draining branches.
# Define the reference block size of the retention curve in centimeters
if GENUCHTEN:
    basic_block_size = 10 / 12  # the reference size of the block for 20/30 sand
else:
    basic_block_size = 1.0  # the reference size of the block for the logistic retention curve

A_WB = (1 / basic_block_size) * dx_PAR
A_DB = (1 / basic_block_size) * dx_PAR

# Output for the terminal.
print("Followed parameters are used for the simulation.")
print(f"Initial saturation [-]:                               {S0}")
print(f"Simulation time [s]:                                  {REALTIME}")
print(f"Size of the block used for simulation [cm]:           {dL * 100}")
print(f"Basic size of the block for the retention curve [cm]: {basic_block_size}")
print(f"Time step [s]:                                        {dt}")
print(f"Width and depth of the medium respectively [m]:       {A},{B}")
print(f"Boundary flux [m/s]:                                  {Q0}")

if GENUCHTEN:
    print("Van Genuchten retention curve is used for the simulation.")
else:
    print("Logistic retention curve is used for the simulation.")

# Memory allocation
S = np.zeros((N, M))  # saturation matrix
Snew = np.zeros((N, M))  # saturation matrix for next iteration
S0_ini = np.ones((N, M))  # initial saturation matrix

# allocation memory for bottom boundary condition defined by residual saturation S_rs
boundFlux = np.zeros((1, M))
boundLim = np.zeros((1, M))

Q1 = np.zeros((N, M + 1))  # flux matrix for side fluxes
Q2 = np.zeros((N + 1, M))  # flux matrix for downward fluxes
Q = np.zeros((N, M))  # flux matrix for/in "each block"

P = np.zeros((N, M))  # pressure matrix
Pwett = np.zeros((N, M))  # pressure for wetting curve
Pdrain = np.zeros((N, M))  # pressure for draining curve
wett = np.zeros((N, M))  # logical variable for wetting mode
drain = np.zeros((N, M))  # logical variable for draining mode

perm = np.zeros((N, M))  # relative permeability

Saturation = np.zeros((N, M, REALTIME))  # Saturation field for saving data
Pressure = np.zeros((N, M, REALTIME))  # Pressure field for saving data

# Fluxes fields for data saving
QQ1 = np.zeros((N, M + 1, REALTIME))  # side fluxes
QQ2 = np.zeros((N + 1, M, REALTIME))  # downward fluxes
QQ = np.zeros((N, M, REALTIME))  # fluxes for/in each block

# Distribution of intrinsic permeability

# Set false if you don't want to have randomization of the intrinsic permeability
RANDOMIZATION_INTRINSIC_PERMEABILITY = True

# Two different methods of randomization of intrinsic permeability: filter and interpolation methods. 
# Interpolation method is recommended. For more details, see:
# Kmec, J.: Analysis of the mathematical models for unsaturated porous media flow, 
METHOD_FILTER = False  # imfilter method  filter method
METHOD_INTERPOLATION = True  # imresize method  interpolation method
KERNEL_SIZE = 6

LOAD_FROM_FILE = False  # If you already have distribution of intrinsic permeability defined in the file

if LOAD_FROM_FILE:
    random_perm = np.load("random_perm.npy")

if not LOAD_FROM_FILE and RANDOMIZATION_INTRINSIC_PERMEABILITY:
    if METHOD_FILTER:
        random_perm = np.random.normal(0, 1, S.shape) * 0.8

        random_perm = cv2.filter2D(random_perm, -1, np.ones((KERNEL_SIZE, KERNEL_SIZE), np.float64) / KERNEL_SIZE)
        np.save("random_perm_filter.npy", random_perm)

    if METHOD_INTERPOLATION:
        # define intrinsic permeability for the blocks of the size 2.5cm
        block_par = 0.025
        index1 = math.ceil(A / block_par)
        index2 = math.ceil(B / block_par)

        interpolationBlocks = block_par / dL
        random_perm = np.random.normal(0, 1, [index2, index1]) * 0.3

        # TODO if debug, to stejné nahoře
        # cv2.imshow("randomPerm1", randomPerm)
        random_perm = cv2.resize(random_perm, None, fx=interpolationBlocks, fy=interpolationBlocks,
                                 interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("randomPerm2", randomPerm)
        # cv2.waitKey(0)
        np.save("random_perm_resize.npy", random_perm)

if RANDOMIZATION_INTRINSIC_PERMEABILITY:
    nasob = np.zeros_like(random_perm)
    nasob[random_perm > 0] = (1 + random_perm[random_perm > 0])
    nasob[random_perm < 0] = (1. / (1 - random_perm[random_perm < 0]))
else:
    nasob = np.ones_like(S)

k_rnd = KAPPA * nasob

print('################# INTRINSIC PERMEABILITY #################')

if RANDOMIZATION_INTRINSIC_PERMEABILITY:
    print(
        f"The minimum and maximum of the field randomPerm respectively:  {np.amin(random_perm)}, {np.amax(random_perm)}")
    print(f"The minimum and maximum of the intrinsic permeablity:          {np.amin(k_rnd)}, {np.amax(k_rnd)}")
    print(f"The average and predefined intrinsic permeablity respectively: {np.mean(k_rnd)}, {KAPPA}")
else:
    print("The distribution of the intrinsic permeability is not used.")

print("#" * 20)

# Initialization of porous media flow

# Initial saturation
S0_ini = S0_ini * S0
S = S0_ini

# Definition of top boundary condition: three possibilities can be chosen.
if FLUX_FULL and not FLUX_MIDDLE:  # Flux q0 at whole top boundary.
    Q2[0, :] = Q0

elif not FLUX_FULL and FLUX_MIDDLE:  # Flux q0 at the middle at 1 cm.
    middle = (A - 0.01) / 2
    pom = round(middle / dL)
    vec = round(0.01 / dL)

    Q2[0, pom: pom + vec] = Q0

else:  # Flux q0 only in the middle block.
    Q2[0, round(M / 2)] = Q0

# Initial capillary pressure. For a parameter
#   - 'wet' we start on the main wetting branch
#   - 'drain' we start on the main draining branch.

retention_curves = RetentionCurves()

match WHICH_BRANCH:
    case "wet":
        if GENUCHTEN:
            P = retention_curves.van_genuchten(S, ALFA_W, N_W, A_WB, RHO, G)
        else:
            P = retention_curves.wet(S, A_WB)

    case "drain":
        if GENUCHTEN:
            P = retention_curves.van_genuchten(S, ALFA_D, N_D, A_DB, RHO, G)
        else:
            P = retention_curves.drain(S, A_DB)

    case _:
        raise Exception("The parameter WHICH_BRANCH is not set correctly.")

time_start = time.time()

# Main part - saturation, pressure and flux update
for k in tqdm(range(iteration)):
    # ---------------SATURATION UPDATE--------------------------------------
    Q[:N, :M] = Q1[:N, :M] - Q1[:N, 1:M + 1] + Q2[:N, :M] - Q2[1:N + 1, :M]
    Snew = S + SM * Q

    # If the flux is too large, then the saturation would increase over unity.
    # In 1D, we simply returned excess water to the block it came from. This
    # approach should be generalized in 2D in such a way that excess water is
    # returned from where it came proportionally to the fluxes. Here we use
    # only the implementation provided for the 1D case. Thus water is returned only above.
    # However, for all the 2D simulations published or are in reviewing process,
    # saturation had never reached unity so this implementation was not used.

    while np.amax(np.abs(Snew)) > LIM_VALUE:
        S_over = np.zeros((N, M))
        S_over[Snew > LIM_VALUE] = Snew(Snew > LIM_VALUE) - LIM_VALUE
        Snew[Snew > LIM_VALUE] = LIM_VALUE

        id1, id2 = np.nonzero(S_over[1:, :] > 0)

        for i in range(len(id1)):
            Snew[id1[i], id2[i]] = Snew[id1[i], id2[i]] + S_over[id1[i] + 1, id2[i]]

    # bottom boundary condition residual saturation is used
    boundLim = np.minimum(Snew[N-1, :], np.ones((1, M)) * SATURATION_RESIDUAL)
    Snew[N-1, :] = np.maximum(Snew[N-1, :] + SM * boundFlux, boundLim)

    # ---------------PRESSURE UPDATE----------------------------------------

    # hysteresis
    P = P + Kps * (Snew - S)

    if GENUCHTEN:
        Pwett = retention_curves.van_genuchten(S, ALFA_W, N_W, A_WB, RHO, G)
        Pdrain = retention_curves.van_genuchten(S, ALFA_D, N_D, A_DB, RHO, G)
    else:
        Pwett = retention_curves.wet(S, A_WB)
        Pdrain = retention_curves.drain(S, A_DB)

    wett = (Snew - S) > 0  # logical matrix for wetting branch
    drain = (S - Snew) > 0  # logical matrix for draining branch

    P[wett] = np.minimum(P[wett], Pwett[wett])
    P[drain] = np.maximum(P[drain], Pdrain[drain])

    # ---------------FLUX UPDATE--------------------------------------------

    # Side fluxes at boundary are set to zero.
    perm = relative_permeability(Snew)

    Q1[:, 1:M] = (1 / MU) * \
        np.sqrt(k_rnd[:N, :M - 1]) * \
        np.sqrt(k_rnd[:N, 1:M]) * \
        np.sqrt(perm[:N, :M - 1]) * \
        np.sqrt(perm[:N, 1:M]) * \
        (0 * RHO * G - ((P[:N, 1:M] - P[:N, :M - 1]) / dL))

    Q2[1:N, :] = (1 / MU) * \
        np.sqrt(k_rnd[:N - 1, :M]) * \
        np.sqrt(k_rnd[1:N, :M]) * \
        np.sqrt(perm[:N - 1, :M]) * \
        np.sqrt(perm[1:N, :M]) * \
        (RHO * G - ((P[1:N, :M] - P[:N - 1, :M]) / dL))
    S = Snew

    # Calculation of flux at bottom boundary.
    boundFlux[0, :] = (1 / MU) * k_rnd[N-1, :M] * perm[N-1, :M] * (RHO * G - ((0 - P[N-1, :M]) / dL))

    # ---------------SAVING DATA--------------------------------------------

    # saving data and check mass balance law
    if k % (RATIO * (1 / dtBase) * TIME_INTERVAL) == 0:
        t = round(k * dt)  # calculation a real simulation time

        Saturation[:, :, t] = S
        Pressure[:, :, t] = P

        QQ1[:, :, t] = Q1
        QQ2[:, :, t] = Q2
        QQ[:, :, t] = Q

        # Check the mass balance law.
        # Implemented only for the case in which the flux at the top
        # boundary is in the middle at 1cm.
        if not FLUX_FULL and FLUX_MIDDLE:
            chyba = (np.sum(S) - np.sum(S0_ini)) - k * SM * Q0 * (1 / dx_PAR)  # absolute error
            relchyba = chyba / (k * SM * Q0)  # relative error

            if abs(relchyba) > 1e-10:
                print(f"Error in saturation: absolute and relative errors respectively {chyba} {relchyba}")

        # Information of calculated simulation time printed on the terminal.
        print(f"Simulation time and real time respectively: {t}, seconds {time.time() - time_start} seconds")

    # Only for a code testing purpose.
    if np.abs(np.amax(S)) > LIM_VALUE:
        raise Exception("Saturation is over limValue defined in the code. Something is wrong.")

print(f"The simulation lasted: {time.time() - time_start} seconds.")

# Data saving
if SAVE_DATA:
    np.save(f"res/dx_{dL}_initial_saturation_{S0}_saturation.npy", Saturation)
    np.save(f"res/dx_{dL}_initial_saturation_{S0}_pressure.npy", Pressure)
    np.save(f"res/dx_{dL}_initial_saturation_{S0}_qq1.npy", QQ1)
    np.save(f"res/dx_{dL}_initial_saturation_{S0}_qq2.npy", QQ2)
    np.save(f"res/dx_{dL}_initial_saturation_{S0}_qq.npy", QQ)

# Time plot in two/three dimensions

if PLOT_TIME:
    step = 10  # time step plot figures
    nn, mm, kk = Saturation.shape

    if mm > 1:  # bar3 plot for 2D semi-continuum model
        # TODO fig = figure('position',[50 50 1100 600])
        mat = np.zeros((nn, mm))
        # TODO colormap('winter')
        # TODO colormap(flipud(colormap))

        for time in np.arange(0, kk, step):
            text = f"Time = {time} [s]   {S0}"
            mat[:, :] = Saturation[:, :, time]
            # TODO hSurface=bar3(mat)
            # plt.title(text)
            # TODO axis([0 mm 0 nn 0 1])
            # TODO view(-120, 40)
            # TODO pause(.001)

    else:  # plot for 1D semi-continuum model
        for time in np.arange(0, kk, step):
            xx = np.arange(dL, B, dL)
            yy = Saturation[:, 0, time]
            # plt.plot(xx, yy, color="k")
            # plt.title(f"Saturation in 1D: Time = {time} [s]   {S0}")
            # TODO axis([0 b 0 1])
            # TODO pause(.001)

# Plot retention curve
# plot the basic retention curve and its linear modification defined by the
# scaling of the retention curve
if PLOT_RETENTION_CURVE:
    SS = np.arange(0.001, 0.999, 0.001)
    Pwett = retention_curves.van_genuchten(SS, ALFA_W, N_W, 1, RHO, G)
    Pdrain = retention_curves.van_genuchten(SS, ALFA_D, N_D, 1, RHO, G)
    plt.plot(SS, Pwett, color="r", label="Basic WB")
    plt.plot(SS, Pdrain, color="k", label="Basic DB")

    P1 = retention_curves.van_genuchten(SS, ALFA_W, N_W, A_WB, RHO, G)
    P2 = retention_curves.van_genuchten(SS, ALFA_D, N_D, A_DB, RHO, G)
    plt.plot(SS, P1, color="r", linestyle="dashed", label="Updated WB: scaled retention curve")
    plt.plot(SS, P2, color="k", linestyle="dashed", label="Updated DB: scaled retention curve")

    plt.legend()
    plt.show()
