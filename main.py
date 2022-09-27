# TODO Q3
# TODO 3D graf
# TODO předpočítat retenční křivky

from retention_curves import *
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from tqdm import tqdm
import seaborn as sns
import numpy as np
import math
import time
import cv2

sns.set_theme()

# Define constants
REALTIME = 1  # simulation time [s]
dL = 0.25 * 0.01  # size of the block [m]
dx_PAR = dL / 0.01  # discretization parameter [-]  ratio of dL and 0.01m
S0 = 0.01  # initial saturation [-]

A = 0.31  # width of the medium [m]  for a=dL 1D semi-continuum model is used
B = 0.50  # depth of the medium [m]
C = A

X = math.floor(A / dL)  # number of blocks in a row
Y = math.floor(B / dL)  # number of blocks in a column
Z = math.floor(C / dL)  # number of blocks in 3D

G = 9.81  # acceleration due to gravity
THETA = 0.35  # porosity
KAPPA = 2.293577981651376e-10  # intrinsic permeability
MU = 9e-4  # dynamic viscosity
RHO = 1000  # density of water

MU_INVERSE = 1 / MU

RHO_G = RHO * G  # density of water times acceleration due to gravity

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
GENUCHTEN = True

# Definition of the initial pressure. You can either start on the main wetting branch or main draining branch
WHICH_BRANCH = "wet"

# van Genuchten parameters for 20/30 sand

M_Q = 1 - 1 / VanGenuchtenWet.N

# Definition of the relative permeability
LAMBDA = 0.8

TIME_INTERVAL = 1.0  # define interval in [s]
LIM_VALUE = 0.999  # instead of unity, the value very close to unity is used

M_Q_inverse = 1 / M_Q

OUTPUT_DIR = "res"

# Definition of the time step
dtBase = 1e-3 * 0.25  # time step [s] for dL=0.01[m]
dt = (dx_PAR ** 2) * dtBase  # time step [s], typical choice of time step parameter for parabolic equation
RATIO = 1. / dx_PAR ** 2  # dtBase/dt=1./dx_par^2
SM = dt / (THETA * dL)  # [s/m] parameter

iteration = round(REALTIME / dt)  # number of iteratioN, t=dt*iter is REALTIME

PLOT_TIME = True  # Set True for time plot of porous media flow
SAVE_DATA = True  # Set True if you want to save Saturation and Pressure data

# Set true if you want to plot the basic retention curve and its linear modification which corresponds to the size of
# the block used for simulation
PLOT_RETENTION_CURVE = True

# DEFINITION OF THE REFERENCE BLOCK SIZE
# The crucial idea of the semi-continuum model is the scaling of the retention curve.
# For more details, see: https://doi.org/10.1038/s41598-022-11437-9

# Parameters A_WB, A_DB define the linear multiplication of the retention curve for the wetting and draining branches.
# Define the reference block size of the retention curve in centimeters
if GENUCHTEN:
    basic_block_size = 10.0 / 12.0  # the reference size of the block for 20/30 sand
else:
    basic_block_size = 1.0  # the reference size of the block for the logistic retention curve

A_RC = (1 / basic_block_size) * dx_PAR

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

# DTYPE = np.longdouble  # = "float128"
DTYPE = np.double  # = "float64"

# Memory allocation
S = np.zeros((Z, Y, X), dtype=DTYPE)  # Saturation matrix
S_new = np.zeros((Z, Y, X), dtype=DTYPE)  # Saturation matrix for next iteration
S0_ini = np.ones((Z, Y, X), dtype=DTYPE)  # Initial saturation matrix

# Bottom boundary condition defined by residual saturation S_rs
bound_flux = np.zeros((Z, 1, X), dtype=DTYPE)
bound_lim = np.zeros((Z, 1, X), dtype=DTYPE)

Q1 = np.zeros((Z, Y, X + 1), dtype=DTYPE)  # Flux matrix for side fluxes
Q2 = np.zeros((Z, Y + 1, X), dtype=DTYPE)  # Flux matrix for downward fluxes
Q3 = np.zeros((Z + 1, Y, X), dtype=DTYPE)  # Flux matrix for 3D Z fluxes
Q = np.zeros((Z, Y, X), dtype=DTYPE)  # Flux matrix for/in "each block"

P = np.zeros((Z, Y, X), dtype=DTYPE)  # Pressure matrix
P_wet = np.zeros((Z, Y, X), dtype=DTYPE)  # Pressure for wetting curve
P_drain = np.zeros((Z, Y, X), dtype=DTYPE)  # Pressure for draining curve
wet = np.zeros((Z, Y, X), dtype=DTYPE)  # Logical variable for wetting mode
drain = np.zeros((Z, Y, X), dtype=DTYPE)  # Logical variable for draining mode

perm = np.zeros((Z, Y, X), dtype=DTYPE)  # relative permeability

saturation = np.zeros((Z, Y, X, REALTIME), dtype=DTYPE)  # Saturation field for saving data
pressure = np.zeros((Z, Y, X, REALTIME), dtype=DTYPE)  # Pressure field for saving data

# Fluxes fields for data saving
QQ1 = np.zeros((Z, Y, X + 1, REALTIME), dtype=DTYPE)  # Side fluxes
QQ2 = np.zeros((Z, Y + 1, X, REALTIME), dtype=DTYPE)  # Downward fluxes
QQ3 = np.zeros((Z + 1, Y + 1, X, REALTIME), dtype=DTYPE)  # Downward fluxes
QQ = np.zeros((Z, Y, X, REALTIME), dtype=DTYPE)  # Fluxes for/in each block

# Distribution of intrinsic permeability - False if you don't want to have randomization of the intrinsic permeability
RANDOMIZATION_INTRINSIC_PERMEABILITY = True

# Two different methods of randomization of intrinsic permeability: filter and interpolation methods. 
# Interpolation method is recommended - Kmec, J.: Analysis of the mathematical models for unsaturated porous media flow
METHOD_FILTER = False  # filter method
METHOD_INTERPOLATION = True  # interpolation method
KERNEL_SIZE = 6

LOAD_FROM_FILE = False  # If you already have distribution of intrinsic permeability defined in the file

if LOAD_FROM_FILE:
    random_perm = np.load("random_perm.npy")

elif RANDOMIZATION_INTRINSIC_PERMEABILITY:
    if METHOD_FILTER:  # TODO 3D
        random_perm = np.random.normal(0, 1, S.shape) * 0.8
        random_perm = cv2.filter2D(random_perm, -1, np.ones((KERNEL_SIZE, KERNEL_SIZE), np.float64) / KERNEL_SIZE)

        # TODO 3D
        # sns.heatmap(random_perm)
        # plt.title("Randomized intrinsic permeability")
        # plt.savefig(f"{OUTPUT_DIR}/random_perm_filter.png")
        # plt.clf()

        np.save("random_perm_filter.npy", random_perm)

    elif METHOD_INTERPOLATION:
        # Define intrinsic permeability for the blocks of the size 2.5cm
        block_par = 0.025

        interpolation_blocks = block_par / dL
        random_perm = np.random.normal(0, 1, [
            math.ceil(C / block_par),
            math.ceil(B / block_par),
            math.ceil(A / block_par)]) * 0.3

        # TODO 3D
        # sns.heatmap(random_perm)
        # plt.title("Randomized intrinsic permeability - before")
        # plt.savefig(f"{OUTPUT_DIR}/random_perm_interpolation_before.png")
        # plt.clf()
        print(random_perm.shape)

        random_perm = zoom(random_perm, (interpolation_blocks, interpolation_blocks, interpolation_blocks))

        # TODO 3D
        # sns.heatmap(random_perm)
        # plt.title("Randomized intrinsic permeability - after")
        # plt.savefig(f"{OUTPUT_DIR}/random_perm_interpolation_after.png")
        # plt.clf()

        np.save(f"{OUTPUT_DIR}/random_perm_interpolation.npy", random_perm)

if RANDOMIZATION_INTRINSIC_PERMEABILITY:
    nasob = np.zeros_like(random_perm)
    nasob[random_perm > 0] = (1 + random_perm[random_perm > 0])
    nasob[random_perm < 0] = (1. / (1 - random_perm[random_perm < 0]))
else:
    nasob = np.ones_like(S)

print(random_perm.shape)
k_rnd = KAPPA * nasob
k_rnd_sqrt = np.sqrt(k_rnd)
k_rnd_q1 = MU_INVERSE * k_rnd_sqrt[:Z, :Y, :X - 1] * k_rnd_sqrt[:Z, :Y, 1:X]
k_rnd_q2 = MU_INVERSE * k_rnd_sqrt[:Z, :Y - 1, :X] * k_rnd_sqrt[:Z, 1:Y, :X]
k_rnd_q3 = MU_INVERSE * k_rnd_sqrt[:Z - 1, :Y, :X] * k_rnd_sqrt[1:Z, :Y, :X]

print("################# INTRINSIC PERMEABILITY #################")
if RANDOMIZATION_INTRINSIC_PERMEABILITY:
    print(f"The minimum of random_perm is {np.amin(random_perm)} and maximum is {np.amax(random_perm)}")
    print(f"The minimum of the intrinsic permeability is {np.amin(k_rnd)} and maximum is {np.amax(k_rnd)}")
    print(f"The average is {np.mean(k_rnd)} and predefined intrinsic permeability respectively is {KAPPA}")
else:
    print("The distribution of the intrinsic permeability is not used.")

print("#" * 20)

# Initialization of porous media flow
# Initial saturation
S0_ini = S0_ini * S0
S = S0_ini

# Definition of top boundary condition: three possibilities can be chosen.
if FLUX_FULL and not FLUX_MIDDLE:  # Flux q0 at whole top boundary.
    Q2[:, 0, :] = Q0

elif not FLUX_FULL and FLUX_MIDDLE:  # Flux q0 at the middle at 1 cm.
    middle_x = (A - 0.01) / 2
    middle_z = (C - 0.01) / 2
    pom_x = round(middle_x / dL)
    pom_z = round(middle_z / dL)
    vec = round(0.01 / dL)

    Q2[pom_z:pom_z, 0, pom_x: pom_x + vec] = Q0

else:  # Flux q0 only in the middle block.
    Q2[round(Z / 2), 0, round(X / 2)] = Q0

retention_curve_wet = (VanGenuchtenWet(A_RC, RHO_G) if GENUCHTEN else RetentionCurveWet(A_RC)).calculate
retention_curve_drain = (VanGenuchtenDrain(A_RC, RHO_G) if GENUCHTEN else RetentionCurveDrain(A_RC)).calculate

# Initial capillary pressure. For a parameter
#   - 'wet' we start on the main wetting branch
#   - 'drain' we start on the main draining branch.
P = retention_curve_wet(S) if WHICH_BRANCH == "wet" else retention_curve_drain(S)

bound_residual = np.ones((Z, 1, X), dtype=DTYPE) * SATURATION_RESIDUAL

time_start = time.time()

# Main part - saturation, pressure and flux update
for k in tqdm(range(1, iteration+1)):
    # --------------- SATURATION UPDATE ---------------
    Q[:Z, :Y, :X] = Q1[:Z, :Y, :X] - Q1[:Z, :Y, 1:X + 1] \
                    + Q2[:Z, :Y, :X] - Q2[:Z, 1:Y + 1, :X] \
                    + Q3[:Z, :Y, :X] - Q3[1:Z + 1, :Y, :X]

    S_new = S + SM * Q

    # If the flux is too large, then the saturation would increase over unity.
    # In 1D, we simply returned excess water to the block it came from. This approach should be generalized in 2D in
    # such a way that excess water is returned from where it came proportionally to the fluxes. Here we use only the
    # implementation provided for the 1D case. Thus water is returned only above. However, for all the 2D simulations
    # published or are in reviewing process, saturation had never reached unity so this implementation was not used.
    while np.amax(np.abs(S_new)) > LIM_VALUE:
        print("Error - that should not happen")
        S_over = np.zeros((Z, Y, X), dtype=DTYPE)
        S_over[S_new > LIM_VALUE] = S_new(S_new > LIM_VALUE) - LIM_VALUE
        S_new[S_new > LIM_VALUE] = LIM_VALUE

        id1, id2, id3 = np.nonzero(S_over[:, 1:, :] > 0)

        for i in range(len(id1)):
            S_new[id1[i], id2[i], id3[i]] = S_new[id1[i], id2[i], id3[i]] + S_over[id1[i], id2[i] + 1, id3[i]]

    # Bottom boundary condition residual saturation is used
    bound_lim = np.minimum(S_new[:, Y - 1:Y, :], bound_residual)
    S_new[:, Y - 1:Y, :] = np.maximum(S_new[:, Y - 1:Y, :] + SM * bound_flux, bound_lim)

    # --------------- PRESSURE UPDATE ---------------
    # Hysteresis
    P = P + Kps * (S_new - S)

    P_wet = retention_curve_wet(S)
    P_drain = retention_curve_drain(S)

    wet = (S_new - S) > 0  # logical matrix for wetting branch
    drain = (S - S_new) > 0  # logical matrix for draining branch

    P[wet] = np.minimum(P[wet], P_wet[wet])
    P[drain] = np.maximum(P[drain], P_drain[drain])

    # --------------- FLUX UPDATE ---------------
    # Calculate relative permeability Side fluxes at boundary are set to zero.
    perm = S_new ** LAMBDA * (1 - (1 - S_new ** M_Q_inverse) ** M_Q) ** 2
    perm_sqrt = np.sqrt(perm)

    Q1[:, :, 1:X] = k_rnd_q1 * \
        perm_sqrt[:Z, :Y, :X - 1] * \
        perm_sqrt[:Z, :Y, 1:X] * \
        (- ((P[:Z, :Y, 1:X] - P[:Z, :Y, :X - 1]) / dL))

    Q2[:, 1:Y, :] = k_rnd_q2 * \
        perm_sqrt[:Z, :Y - 1, :X] * \
        perm_sqrt[:Z, 1:Y, :X] * \
        (RHO_G - ((P[:Z, 1:Y, :X] - P[:Z, :Y - 1, :X]) / dL))

    Q3[1:Z, :, :] = k_rnd_q3 * \
        perm_sqrt[:Z - 1, :Y, :X] * \
        perm_sqrt[1:Z, :Y, :X] * \
        (RHO_G - ((P[1:Z, :Y, :X] - P[:Z - 1, :Y, :X]) / dL))

    S = S_new

    # Calculation of flux at bottom boundary.
    bound_flux[:, 0, :] = MU_INVERSE * k_rnd[:Z, Y - 1, :X] * perm[:Z, Y - 1, :X] * (RHO_G - ((0 - P[:Z, Y - 1, :X]) / dL))

    # --------------- Saving data and check mass balance law ---------------
    if k % (RATIO * (1 / dtBase) * TIME_INTERVAL) == 0:
        t = round(k * dt) - 1  # calculation a real simulation time
        saturation[:, :, :, t] = S
        pressure[:, :, :, t] = P

        QQ1[:, :, :, t] = Q1
        QQ2[:, :, :, t] = Q2
        QQ3[:, :, :, t] = Q3
        QQ[:, :, :, t] = Q

        # Check the mass balance law
        # Implemented only for the case in which the flux at the top boundary is in the middle at 1cm
        if not FLUX_FULL and FLUX_MIDDLE:
            error = (np.sum(S) - np.sum(S0_ini)) - k * SM * Q0 * (1 / dx_PAR)  # absolute error
            error_relative = error / (k * SM * Q0)  # relative error

            # if abs(error_relative) > 1e-10:
            print(f"Error in saturation:\n\t- absolute:\t {error}\n\t- relative:\t{error_relative}")

        # Information of calculated simulation time printed on the terminal.
        print(f"Simulation time is {t+1} s, simulation is running for {time.time() - time_start} s")

    # Only for a code testing purpose.
    if np.abs(np.amax(S)) > LIM_VALUE:
        raise Exception(f"Saturation is over {LIM_VALUE} defined in the code.")

print(f"The simulation lasted: {time.time() - time_start} s")

# Data saving
if SAVE_DATA:
    np.save(f"{OUTPUT_DIR}/dx_{dL}_initial_saturation_{S0}_saturation.npy", saturation)
    np.save(f"{OUTPUT_DIR}/dx_{dL}_initial_saturation_{S0}_pressure.npy", pressure)
    np.save(f"{OUTPUT_DIR}/dx_{dL}_initial_saturation_{S0}_qq1.npy", QQ1)
    np.save(f"{OUTPUT_DIR}/dx_{dL}_initial_saturation_{S0}_qq2.npy", QQ2)
    np.save(f"{OUTPUT_DIR}/dx_{dL}_initial_saturation_{S0}_qq.npy", QQ)

# Time plot in two/three dimensions
if PLOT_TIME:
    step = 10  # time step plot figures
    nn, mm, kk = saturation.shape

    if mm > 1:  # bar3 plot for 2D semi-continuum model
        # TODO fig = figure('position',[50 50 1100 600])
        mat = np.zeros((nn, mm), dtype=DTYPE)
        # TODO colormap('winter')
        # TODO colormap(flipud(colormap))

        for time in np.arange(0, kk, step):
            text = f"Time = {time} [s]   {S0}"
            mat[:, :] = saturation[:, :, time]
            # TODO hSurface=bar3(mat)
            # plt.title(text)
            # TODO axis([0 mm 0 nn 0 1])
            # TODO view(-120, 40)
            # TODO pause(.001)

    else:  # plot for 1D semi-continuum model
        for time in np.arange(0, kk, step):
            xx = np.arange(dL, B, dL)
            yy = saturation[:, 0, time]
            # plt.plot(xx, yy, color="k")
            # plt.title(f"Saturation in 1D: Time = {time} [s]   {S0}")
            # TODO axis([0 b 0 1])
            # TODO pause(.001)

# Plot the basic retention curve and its linear modification defined by the scaling of the retention curve
if PLOT_RETENTION_CURVE:
    SS = np.arange(0.001, 0.999, 0.001)
    P_wet = VanGenuchtenWet(1, RHO_G).calculate(SS)
    P_drain = VanGenuchtenDrain(1, RHO_G).calculate(SS)
    plt.plot(SS, P_wet, color="r", label="Basic WB")
    plt.plot(SS, P_drain, color="k", label="Basic DB")

    P1 = VanGenuchtenWet(A_RC, RHO_G).calculate(SS)
    P2 = VanGenuchtenDrain(A_RC, RHO_G).calculate(SS)
    plt.plot(SS, P1, color="r", linestyle="dashed", label="Updated WB: scaled retention curve")
    plt.plot(SS, P2, color="k", linestyle="dashed", label="Updated DB: scaled retention curve")

    plt.title("Retention curves")
    plt.legend()
    plt.show()
