import hw
import numpy as np
import env_ingredient_add as env
from pydrake.trajectories import PiecewisePose
from pydrake.systems.framework import Diagram, Context
from pydrake.math import RigidTransform, RotationMatrix
from dataclasses import dataclass
from pydrake.all import PiecewisePolynomial, PiecewisePose, Trajectory, LeafSystem
from enum import Enum
from pydrake.all import (
    AbstractValue,
    BasicVector,
    Context,
    Diagram,
    DiagramBuilder,
    DiscreteValues,
    InputPortIndex,
    LeafSystem,
    MultibodyPlant,
    PathParameterizedTrajectory,
    PiecewisePose,
    PortSwitch,
    RigidTransform,
    State,
)

class PoseTrajectorySource(LeafSystem):
    """
    returns desired list of poses of dimension 20: 10 positions, 10 velocities
    (optional) pose_trajectory: trajectory to follow. if context does not already exist, pass it in from the plant.
    """
    # pose_trajectory: PiecewisePose = PiecewisePose()

    # need a way to set a new trajectory within this method?

    def __init__(self, pose_trajectory):
        LeafSystem.__init__(self)
        self._pose_trajectory = pose_trajectory
        self.DeclareAbstractOutputPort(
            "pose", lambda: AbstractValue.Make(RigidTransform()), self.CalcPose
        )

    def CalcPose(self, context, output):
        output.set_value(self._pose_trajectory.GetPose(context.get_time()))
        pose = self._pose_trajectory.GetPose(context.get_time())
        # print(f"Pose dimensions: {pose.GetAsVector().size()}")
        output.set_value(pose)

# def traj_linear_move_to_bowl_0(
#         diagram,
#         context=None,
#         move_time=10 # seconds
#     ) -> PoseTrajectorySource:
#     if context is None:
#         context = diagram.CreateDefaultContext()

#     start_time = context.get_time()

#     # get a starting point
#     X_WGinit = hw.get_X_WG(diagram,context)

#     # intermediate #1
#     int_1 = hw.RigidTransform(
#         R=hw.RotationMatrix(),
#         p=np.array([0,1,1])
#     )

#     # intermediate #2
#     int_2 = hw.RigidTransform(
#         R=hw.RotationMatrix(),
#         p=np.array([-4,1,1])
#     )

#     # define a goal pose
#     goal_pose = hw.RigidTransform(
#         R=hw.RotationMatrix(),
#         p=np.array(env.bowl_0) + np.array([0.25, 0, 0.25])
#     )

#     poses = hw.PiecewisePose.MakeLinear([start_time,
#         start_time + move_time/4,
#         start_time + 3*move_time/4,
#         start_time + move_time],
#         [X_WGinit, int_1, int_2, goal_pose])

#     return poses


@dataclass
class TrajectoryWithTimingInformation:
    """
    Contains a Drake Trajectory and an associated start time which is relative to
    the simulation start time.
    """

    trajectory: Trajectory = PiecewisePolynomial()
    start_time_s: float = np.nan

@dataclass
class PiecewisePoseWithTimingInformation:
    """
    Contains a Drake PiecewisePose trajectory and an associated start time which is
    relative to the simulation start time.
    """

    trajectory: PiecewisePose = PiecewisePose()
    start_time_s: float = np.nan

@dataclass
class OpenLoopPlanarPushingPlanarTimingInformation:
    """
    Timing information for OpenLoopPlanarPushingPlanar. All times are relative to the
    simulation start and in seconds. All values are NaN by default.
    """

    start_move_to_start: float = np.nan
    """Start time of the move to pushing start trajectory."""
    end_move_to_start: float = np.nan
    """Finish time of the move to pushing start trajeectory."""
    start_pushing: float = np.nan
    """Start time of the pushing trajectory."""
    end_pushing: float = np.nan
    """End time of the pushing trajectory."""


class PizzaRobotState(Enum):
    """FSM state enumeration"""

    START = 0
    PLAN_IIWA_PAINTER = 1
    EXECUTE_IIWA_PAINTER = 2
    FINISHED = 3


class PizzaPlanner(LeafSystem):
    def __init__(self, 
        num_joint_positions: int,
        initial_delay_s: int,
        controller_plant: MultibodyPlant
    ):
        super().__init__()
        self._is_finished = False
        self._iiwa_plant = controller_plant
        self._fsm_state_idx = int(
            self.DeclareAbstractState(AbstractValue.Make(PizzaRobotState.START))
        )
        self._timing_information_idx = int(
            self.DeclareAbstractState(
                AbstractValue.Make(
                    OpenLoopPlanarPushingPlanarTimingInformation(
                        start_move_to_start=initial_delay_s
                    )
                )
            )
        )
        self._current_iiwa_positions_idx = int(
            self.DeclareDiscreteState(num_joint_positions)
        )
        self._current_pose_trajectory_idx = int(
            self.DeclareAbstractState(
                AbstractValue.Make(PiecewisePoseWithTimingInformation())
            )
        )
        self._iiwa_state_estimated_input_port = self.DeclareVectorInputPort(
            "mobile_iiwa.estimated_state", num_joint_positions * 2
        )
        self.DeclareVectorOutputPort(
            "current_iiwa_positions", num_joint_positions, self._get_current_iiwa_positions
        )
        self.DeclareInitializationDiscreteUpdateEvent(self._initialize_discrete_state)
        self.DeclarePerStepUnrestrictedUpdateEvent(self._run_fsm_logic)

    def _get_current_iiwa_positions(self, context: Context, output: BasicVector) -> None:
        positions = context.get_discrete_state(self._current_iiwa_positions_idx).get_value()
        output.set_value(positions)

    def _initialize_discrete_state(self, context: Context, discrete_values: DiscreteValues) -> None:
        discrete_values.set_value(
            self._current_iiwa_positions_idx,
            self._iiwa_state_estimated_input_port.Eval(context)[:10],
        )

    def _run_fsm_logic(self, context: Context, state: State) -> None:
        current_time = context.get_time()
        timing_information: OpenLoopPlanarPushingPlanarTimingInformation = (
            context.get_mutable_abstract_state(self._timing_information_idx).get_value()
        )
        mutable_fsm_state = state.get_mutable_abstract_state(self._fsm_state_idx)
        fsm_state_value: PizzaRobotState = context.get_abstract_state(self._fsm_state_idx).get_value()

        if fsm_state_value == PizzaRobotState.START:
            print("Current state: START")
            q_current = self._iiwa_state_estimated_input_port.Eval(context)[:10]
            state.get_mutable_discrete_state().set_value(self._current_iiwa_positions_idx, q_current)
            print("Transitioning to PLAN_IIWA_PAINTER FSM state.")
            mutable_fsm_state.set_value(PizzaRobotState.PLAN_IIWA_PAINTER)

        elif fsm_state_value == PizzaRobotState.PLAN_IIWA_PAINTER:
            print("Current state: PLAN_IIWA_PAINTER")
            pose_traj = self.traj_linear_move_to_bowl_0(context)
            state.get_mutable_abstract_state(self._current_pose_trajectory_idx).set_value(
                PiecewisePoseWithTimingInformation(trajectory=pose_traj, start_time_s=current_time)
            )
            print("Transitioning to EXECUTE_IIWA_PAINTER FSM state.")
            mutable_fsm_state.set_value(PizzaRobotState.EXECUTE_IIWA_PAINTER)

        elif fsm_state_value == PizzaRobotState.EXECUTE_IIWA_PAINTER:
            print("Current state: EXECUTE_IIWA_PAINTER")
            pose_trajectory_info: PiecewisePoseWithTimingInformation = (
                context.get_abstract_state(self._current_pose_trajectory_idx).get_value()
            )
            pose_trajectory = pose_trajectory_info.trajectory
            start_time_s = pose_trajectory_info.start_time_s
            if current_time >= start_time_s:
                pose = pose_trajectory.GetPose(current_time - start_time_s)
                state.get_mutable_discrete_state().set_value(self._current_iiwa_positions_idx, pose.translation())
            print("Transitioning to FINISHED FSM state.")
            mutable_fsm_state.set_value(PizzaRobotState.FINISHED)

    def traj_linear_move_to_bowl_0(self, context: Context, move_time=10) -> PiecewisePose:
        if not context.is_root_context():
            context = self.GetMyContextFromRoot(context)
        start_time = context.get_time()
        context_plant = self._iiwa_plant.GetMyContextFromRoot(context)
        X_WGinit = plant.GetFreeBodyPose(context_plant, plant.GetBodyByName("gripper"))
        int_1 = hw.RigidTransform(R=hw.RotationMatrix(), p=np.array([0, 1, 1]))
        int_2 = hw.RigidTransform(R=hw.RotationMatrix(), p=np.array([-4, 1, 1]))
        goal_pose = hw.RigidTransform(R=hw.RotationMatrix(), p=np.array(env.bowl_0) + np.array([0.25, 0, 0.25]))
        poses = hw.PiecewisePose.MakeLinear(
            [start_time, start_time + move_time / 4, start_time + 3 * move_time / 4, start_time + move_time],
            [X_WGinit, int_1, int_2, goal_pose]
        )
        return poses

# class PizzaPlanner(LeafSystem):
#     def __init__(
#             self,
#             num_joint_positions: int,
#             initial_delay_s: int
#         ):
#         """
#         Args: 
#             (blank for now)

#         Adapted from OpenLoopPlanarPushingPlanner (github user nepfaff/iiwa_setup).
#         """
#         super().__init__()

#         self._is_finished = False # encodes that we've reached goal state

#         # internal state
#         self._fsm_state_idx = int(
#             self.DeclareAbstractState(
#                 AbstractValue.Make(PizzaRobotState.START) # init to start state
#             )
#         )
#         """current FSM state"""

#         self._timing_information_idx = int(
#             self.DeclareAbstractState(
#                 AbstractValue.Make(
#                     OpenLoopPlanarPushingPlanarTimingInformation(
#                         start_move_to_start=initial_delay_s
#                     )
#                 )
#             )
#         )
#         """Class for writing and reading planed timings."""

#         self._current_iiwa_positions_idx = int(
#             self.DeclareDiscreteState(num_joint_positions)
#         )
#         """The current iiwa positions. These are used to command the robot to stay idle."""

#         self._current_pose_trajectory_idx = int(
#             self.DeclareAbstractState(
#                 AbstractValue.Make(PiecewisePoseWithTimingInformation())
#             )
#         )
#         """The current pose trajectory."""


#         # input ports
#         self._iiwa_state_estimated_input_port = self.DeclareVectorInputPort(
#             "mobile_iiwa.estimated_state", num_joint_positions*2 # positions and velocities
#         )

#         # output ports
#         # self.DeclareAbstractOutputPort(
#         #     "control_mode",
#         #     lambda: AbstractValue.Make(InputPortIndex(0)),
#         #     self._calc_control_mode,
#         # )
#         # self.DeclareAbstractOutputPort(
#         #     "reset_diff_ik",
#         #     lambda: AbstractValue.Make(False),
#         #     self._calc_diff_ik_reset,
#         # )
#         # self.DeclareAbstractOutputPort(
#         #     "joint_position_trajectory",
#         #     lambda: AbstractValue.Make(TrajectoryWithTimingInformation()),
#         #     self._get_current_joint_position_trajectory,
#         # )
#         # self.DeclareAbstractOutputPort(
#         #     "pose_trajectory",
#         #     lambda: AbstractValue.Make(PiecewisePoseWithTimingInformation()),
#         #     self._get_current_pose_trajectory,
#         # )
#         self.DeclareVectorOutputPort(
#             "current_iiwa_positions",
#             num_joint_positions,
#             self._get_current_iiwa_positions,
#         )

#         self.DeclareInitializationDiscreteUpdateEvent(self._initialize_discrete_state)
       
#         # Run FSM logic before every trajectory-advancing step
#         self.DeclarePerStepUnrestrictedUpdateEvent(self._run_fsm_logic)


#     def _get_current_iiwa_positions(
#         self, context: Context, output: BasicVector
#     ) -> None:
#         positions = context.get_discrete_state(
#             self._current_iiwa_positions_idx
#         ).get_value()
#         output.set_value(positions)


#     def _initialize_discrete_state(
#         self, context: Context, discrete_values: DiscreteValues
#     ) -> None:
#         # Initialize the current iiwa positions
#         discrete_values.set_value(
#             self._current_iiwa_positions_idx,
#             self._iiwa_state_estimated_input_port.Eval(context)[:10], # pick off positions
#         )

#     # plan move to home location
#     def _plan_iiwa_painter(self,context: Context) -> PathParameterizedTrajectory:
#         pass

#     # iiwa painter trajectory
#     def _execute_iiwa_painter(self, context: Context):
#         pass


#     def _run_fsm_logic(self, context: Context, state: State) -> None:
#         """FSM state transition logic."""
#         current_time = context.get_time()
#         timing_information: OpenLoopPlanarPushingPlanarTimingInformation = (
#             context.get_mutable_abstract_state(self._timing_information_idx).get_value()
#         )

#         mutable_fsm_state = state.get_mutable_abstract_state(self._fsm_state_idx)
#         fsm_state_value: PizzaRobotState = context.get_abstract_state(
#             self._fsm_state_idx
#         ).get_value()

#         if fsm_state_value == PizzaRobotState.START:
#             print("Current state: START")
#             # for now, this state just inits the robot position at the start time.
#             q_current = self._iiwa_state_estimated_input_port.Eval(context)[:10] # xyz position and joint angles
#             state.get_mutable_discrete_state().set_value(
#                 self._current_iiwa_positions_idx,
#                 q_current,
#             )

#             print("Transitioning to PLAN_IIWA_PAINTER FSM state.")
#             mutable_fsm_state.set_value(PizzaRobotState.PLAN_IIWA_PAINTER)

#             pose_traj = traj_linear_move_to_bowl_0(diagram, context, move_time=10)
#             state.get_mutable_abstract_state(self._current_pose_trajectory_idx).set_value(
#                 PiecewisePoseWithTimingInformation(
#                     trajectory=pose_traj,
#                     start_time_s=current_time
#                 )
#             )

#         elif fsm_state_value == PizzaRobotState.PLAN_IIWA_PAINTER:
#             print("Current state: PLAN_IIWA_PAINTER")
            
#             pose_trajectory_info: PiecewisePoseWithTimingInformation = (
#                 context.get_abstract_state(self._current_pose_trajectory_idx).get_value()
#             )
#             pose_trajectory = pose_trajectory_info.trajectory
#             start_time_s = pose_trajectory_info.start_time_s

#             if current_time >= start_time_s:
#                 # Execute the trajectory
#                 pose = pose_trajectory.GetPose(current_time - start_time_s)
#                 # Update the robot's position to follow the trajectory
#                 discrete_values.set_value(
#                     self._current_iiwa_positions_idx,
#                     pose.translation()
#                 )
#             # this is where we should figure out the trajectory to move.


#             # q_traj = self._plan_move_to_start(context)
#             # state.get_mutable_abstract_state(self._current_joint_traj_idx).set_value(
#             #     TrajectoryWithTimingInformation(
#             #         trajectory=q_traj,
#             #         start_time_s=timing_information.start_move_to_start,
#             #     )
#             # )
#             # timing_information.end_move_to_start = (
#             #     timing_information.start_move_to_start + q_traj.end_time()
#             # )
#             # state.get_mutable_abstract_state(self._timing_information_idx).set_value(
#             #     timing_information
#             # )

#             print("Transitioning to EXECUTE_IIWA_PAINTER FSM state.")
#             mutable_fsm_state.set_value(PizzaRobotState.EXECUTE_IIWA_PAINTER)

#         elif fsm_state_value == PizzaRobotState.EXECUTE_IIWA_PAINTER:
#             # this is where we would actually execute the trajectory

#             # traj0 = hw.create_painter_trajectory(diagram,meshcat,context)

#             print("Transitioning to FINISHED FSM state")
#             mutable_fsm_state.set_value(PizzaRobotState.FINISHED)


#         elif fsm_state_value == PizzaRobotState.FINISHED:

#             self._is_finished = True

#         else:
#             print(f"Invalid FSM state: {fsm_state_value}")
#             exit(1)

#     def is_finished(self) -> bool:
#         """Returns True if the task has been completed and False otherwise."""
#         return self._is_finished


