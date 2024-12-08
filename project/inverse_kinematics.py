import logging

from typing import Optional

import numpy as np

from pydrake.all import (
    InverseKinematics,
    MultibodyPlant,
    RigidTransform,
    RotationMatrix,
    Solve,
    eq
)

# CREDIT: nepfaff/iiwa-setup

def solve_global_inverse_kinematics(
    plant: MultibodyPlant,
    X_G: RigidTransform,
    initial_guess: np.ndarray,
    position_tolerance: float,
    orientation_tolerance: float,
    gripper_frame_name: str = "body",
    fix_base=[False,False,False],
) -> Optional[np.ndarray]:
    """Computes global IK.

    Args:
        plant (MultibodyPlant): The robot control plant.
        X_G (RigidTransform): Gripper pose to compute the joint angles for.
        initial_guess (np.ndarray): The initial guess to use of shape (N,) where N are
        the number of joint positions.
        position_tolerance (float): The position tolerance to use for the global IK
        optimization problem.
        orientation_tolerance (float): The orientation tolerance to use for the global
        IK optimization problem.
        gripper_frame_name (str): The name of the gripper frame.

    Returns:
        Optional[np.ndarray]: Joint positions corresponding to X_G. Returns None if no
        IK solution could be found.
    """
    ik = InverseKinematics(plant)
    q_variables = ik.q()[:12]

    gripper_frame = plant.GetFrameByName(gripper_frame_name)

    # Position constraint
    p_G_ref = X_G.translation()
    ik.AddPositionConstraint(
        frameB=gripper_frame,
        p_BQ=np.zeros(3),
        frameA=plant.world_frame(),
        p_AQ_lower=p_G_ref - position_tolerance,
        p_AQ_upper=p_G_ref + position_tolerance,
    )

    # Orientation constraint
    R_G_ref = X_G.rotation()
    ik.AddOrientationConstraint(
        frameAbar=plant.world_frame(),
        R_AbarA=R_G_ref,
        frameBbar=gripper_frame,
        R_BbarB=RotationMatrix(),
        theta_bound=orientation_tolerance,
    )

    prog = ik.prog()
    # print(f"{q_variables.shape=}")
    # print(f"{initial_guess.shape=}")

    # initial guess is q_current of the robot
    # fixes the base if the axis is fixed
    for i,axis_fixed in enumerate(fix_base):
        if axis_fixed:
            # print(q_variables[i])
            # print(initial_guess[i])
            prog.AddConstraint(q_variables[i] == initial_guess[i])
    
    # add min/max bounding box constraints for x/y work area
    xy_min = np.array([-2.25, +0.25])
    xy_max = np.array([0.25, 2.25])
    prog.AddBoundingBoxConstraint(xy_min, xy_max, q_variables[:2])
    
    prog.SetInitialGuess(q_variables, initial_guess)

    result = Solve(prog)
    if not result.is_success():
        logging.error(f"Failed to solve global IK for gripper pose {X_G}.")
        return None
    q_sol = result.GetSolution(q_variables)
    return q_sol
