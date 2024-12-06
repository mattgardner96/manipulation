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
from manipulation.meshcat_utils import AddMeshcatTriad

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
class PizzaBotTimingInformation:
    """
    Timing information for OpenLoopPlanarPushingPlanar. All times are relative to the
    simulation start and in seconds. All values are NaN by default.
    """

    start_iiwa_painter: float = np.nan
    """Start time of the move to pushing start trajectory."""
    end_iiwa_painter: float = np.nan
    """Finish time of the move to pushing start trajeectory."""
    start_pushing: float = np.nan
    """Start time of the pushing trajectory."""
    end_pushing: float = np.nan
    """End time of the pushing trajectory."""


class PizzaRobotState(Enum):
    """FSM state enumeration"""

    START = 0
    PLAN_IIWA_PAINTER = 1
    PLAN_TRAJ_0 = 2
    EXECUTE_IIWA_PAINTER = 3
    EXECUTE_TRAJECTORY_0 = 4
    FINISHED = 5


class PizzaPlanner(LeafSystem):
    def __init__(self, 
        num_joint_positions: int,
        initial_delay_s: int,
        controller_plant: MultibodyPlant
    ):
        super().__init__()
        self._is_finished = False
        self._iiwa_plant = controller_plant
        self._gripper_frame = self._iiwa_plant.GetFrameByName("body")
        gripper_body = self._iiwa_plant.GetBodyByName("body")
        self._gripper_body_index = gripper_body.index()

        plant_context = self._iiwa_plant.CreateDefaultContext()
        # print(f"within planner: {self._iiwa_plant.GetPositions(plant_context)}")

        self._fsm_state_idx = int(
            self.DeclareAbstractState(AbstractValue.Make(PizzaRobotState.START))
        )
        self._timing_information_idx = int(
            self.DeclareAbstractState(
                AbstractValue.Make(
                    PizzaBotTimingInformation(
                        start_iiwa_painter=initial_delay_s
                    )
                )
            )
        )
        self._current_iiwa_positions_idx = int(
            self.DeclareDiscreteState(num_joint_positions)
        )
        self._pose_trajectory_idx = int(
            self.DeclareAbstractState(
                AbstractValue.Make(PiecewisePoseWithTimingInformation())
            )
        )

        # INPUT PORTS
        self._iiwa_state_estimated_input_port = self.DeclareVectorInputPort(
            "mobile_iiwa.estimated_state", num_joint_positions * 2
        ) # must be first

        self._body_poses_input_port = self.DeclareAbstractInputPort(
            "body_poses",
            AbstractValue.Make([RigidTransform()])
        )

        # Get the initial joint positions from the estimated state input port
        # Note: Since this is during initialization, you might need to set initial positions manually
        initial_iiwa_state = self._iiwa_plant.GetPositions(plant_context)
        if self._iiwa_state_estimated_input_port.HasValue(self.CreateDefaultContext()):
            initial_iiwa_state = self._iiwa_state_estimated_input_port.Eval(self.CreateDefaultContext())
        q0 = initial_iiwa_state[:len(initial_iiwa_state)]

        # Set the positions in the plant context
        self._iiwa_plant.SetPositions(plant_context, q0)

        # Get the gripper pose
        gripper_frame = self._iiwa_plant.GetFrameByName("body")
        default_gripper_pose = self._iiwa_plant.CalcRelativeTransform(
            plant_context,
            self._iiwa_plant.world_frame(),
            gripper_frame
        )

        # print(f"{default_gripper_pose=}")

        self._desired_pose_idx = self.DeclareAbstractState(
            AbstractValue.Make(default_gripper_pose)
        )

        # OUTPUT PORTS (remain the same)
        self.DeclareAbstractOutputPort(
            "desired_pose",
            alloc=lambda: AbstractValue.Make(RigidTransform()),
            calc=self._set_desired_pose
        )

        self.DeclareVectorOutputPort(
            "current_iiwa_positions", num_joint_positions, self._get_current_iiwa_positions
        )

        self.DeclareAbstractOutputPort(
            "reset_diff_ik",
            lambda: AbstractValue.Make(False),
            self._calc_diff_ik_reset,
        )

        self.DeclareInitializationDiscreteUpdateEvent(self._initialize_discrete_state)
        self.DeclarePerStepUnrestrictedUpdateEvent(self._run_fsm_logic)

    def _get_current_iiwa_positions(self, context: Context, output: BasicVector) -> None:
        positions = context.get_discrete_state(self._current_iiwa_positions_idx).get_value()
        output.set_value(positions)

    def _solve_gripper_pose(self, context: Context) -> RigidTransform:
        body_poses = self._body_poses_input_port.Eval(context)
        gripper_pose = body_poses[self._gripper_body_index]
        
        return gripper_pose

    def _set_desired_pose(self, context: Context, output: AbstractValue) -> None:
        desired_pose = context.get_abstract_state(self._desired_pose_idx).get_value()
        output.set_value(desired_pose)

    def _calc_diff_ik_reset(self, context: Context, output: AbstractValue) -> None:
        # credit: used with permission from Nicholas.
        """Logic for deciding when to reset the differential IK controller state."""
        state = context.get_abstract_state(int(self._fsm_state_idx)).get_value()
        if state == PizzaRobotState.PLAN_IIWA_PAINTER:
            # Need to reset before executing a trajectory.
            output.set_value(True)
        else:
            output.set_value(False)

    # init function
    def _initialize_discrete_state(self, context: Context, discrete_values: DiscreteValues) -> None:
        discrete_values.set_value(
            self._current_iiwa_positions_idx,
            self._iiwa_state_estimated_input_port.Eval(context)[:10],
        )

    ###------------ STATE MACHINE LOGIC -----------###
    def _run_fsm_logic(self, context: Context, state: State) -> None:
        current_time = context.get_time()
        timing_information: PizzaBotTimingInformation = (
            context.get_mutable_abstract_state(self._timing_information_idx).get_value()
        )
        mutable_fsm_state = state.get_mutable_abstract_state(self._fsm_state_idx)
        fsm_state_value: PizzaRobotState = context.get_abstract_state(self._fsm_state_idx).get_value()

        if fsm_state_value == PizzaRobotState.START:
            print("Current state: START")
            # q_current = self._iiwa_state_estimated_input_port.Eval(context)[:10]
            # state.get_mutable_discrete_state().set_value(self._current_iiwa_positions_idx, q_current)
            # print("Transitioning to PLAN_IIWA_PAINTER FSM state.")
            # mutable_fsm_state.set_value(PizzaRobotState.PLAN_IIWA_PAINTER)
        
        elif fsm_state_value == PizzaRobotState.PLAN_IIWA_PAINTER:
            gripper_pose = self._solve_gripper_pose(context)

            pose_traj = self._create_painter_traj(
                gripper_pose,
                start_time_s=timing_information.start_iiwa_painter,
                context=context
            )

            state.get_mutable_abstract_state(self._pose_trajectory_idx).set_value(pose_traj)

            # # mutable_fsm_state.set_value(PizzaRobotState.EXECUTE_IIWA_PAINTER)
            # print("Transitioning to EXECUTE_IIWA_PAINTER FSM state.")

        elif fsm_state_value == PizzaRobotState.PLAN_TRAJ_0:
            pass
            # DEADC0DE
            # print("Current state: PLAN_IIWA_PAINTER")
            # pose_traj = self.traj_linear_move_to_bowl_0(context)
            # state.get_mutable_abstract_state(self._current_pose_trajectory_idx).set_value(
            #     PiecewisePoseWithTimingInformation(trajectory=pose_traj, start_time_s=current_time)
            # )
            # print("Transitioning to EXECUTE_IIWA_PAINTER FSM state.")
            # mutable_fsm_state.set_value(PizzaRobotState.EXECUTE_IIWA_PAINTER)

        elif fsm_state_value == PizzaRobotState.EXECUTE_IIWA_PAINTER:

            pose_traj = context.get_abstract_state(self._pose_trajectory_idx).get_value()

            if current_time >= timing_information.start_iiwa_painter:
                desired_pose = pose_traj.trajectory.GetPose(current_time - timing_information.start_iiwa_painter)
                state.get_mutable_abstract_state(self._desired_pose_idx).set_value(desired_pose)
            if current_time >= pose_traj.start_time_s + pose_traj.trajectory.end_time():
                print("Transitioning to FINISHED FSM state.")
                mutable_fsm_state.set_value(PizzaRobotState.FINISHED)
            pass

        elif fsm_state_value == PizzaRobotState.EXECUTE_TRAJECTORY_0:
            traj_info = context.get_abstract_state(
                self._current_pose_trajectory_idx
            ).get_value()

            desired_time = current_time - traj_info.start_time_s
            if desired_time >= traj_info.trajectory.end_time():
                # Trajectory finished
                # Set final pose
                final_pose = traj_info.trajectory.GetPose(traj_info.trajectory.end_time())
                # state.get_mutable_discrete_state().get_mutable_vector(
                #     self._current_iiwa_positions_idx
                # ).set_value(final_pose.translation())
                state.get_mutable_abstract_state(self._desired_pose_idx).set_value(final_pose)
                
                print("Transitioning to FINISHED FSM state.")
                mutable_fsm_state.set_value(PizzaRobotState.FINISHED)
            else:
                # During trajectory execution, update positions based on trajectory
                if desired_time >= 0:
                    pose = traj_info.trajectory.GetPose(desired_time)
                    # Pass the desired pose to the external differential inverse kinematics engine
                    desired_pose = pose
                    # q_current = state.get_discrete_state(self._current_iiwa_positions_idx).get_value()
                else:
                    # Before trajectory start, set to initial pose
                    initial_pose = traj_info.trajectory.GetPose(0.0)
                    state.get_mutable_discrete_state().get_mutable_vector(
                        self._current_iiwa_positions_idx
                    ).set_value(initial_pose.translation())


    def traj_linear_move_to_bowl_0(self, context: Context, move_time=10) -> PiecewisePose:
        start_time = context.get_time()
        # Retrieve the desired pose (implementation depends on your trajectory management)
        # For example purposes, creating dummy poses
        X_WGinit = RigidTransform()
        int_1 = RigidTransform(RotationMatrix(), np.array([0, 1, 1]))
        int_2 = RigidTransform(RotationMatrix(), np.array([-4, 1, 1]))
        goal_pose = RigidTransform(
            R=RotationMatrix(), 
            p=np.array(env.bowl_0) + np.array([0.25, 0, 0.25])
        )
        poses = PiecewisePose.MakeLinear(
            [start_time, 
             start_time + move_time / 4, 
             start_time + 3 * move_time / 4, 
             start_time + move_time],
            [X_WGinit, int_1, int_2, goal_pose]
        )
        return poses

    def _create_painter_traj(
        self,
        X_WG,
        start_time_s=0.0,
        meshcat=None,
        context=None
    ) -> PiecewisePoseWithTimingInformation:
        def compose_circular_key_frames(thetas, X_WCenter, radius):
            """
            returns: a list of RigidTransforms
            """
            # this is an template, replace your code below
            key_frame_poses_in_world = []
            for theta in thetas:
                position = X_WCenter.translation() + np.array([
                    radius * np.cos(theta),
                    radius * np.sin(theta),
                    0.0  # z-coordinate stays constant for a horizontal trajectory
                ])
                
                # Use the same rotation matrix for all keyframes
                rotation_matrix = X_WCenter.rotation()
                this_pose = RigidTransform(rotation_matrix, position)
                key_frame_poses_in_world.append(this_pose)
                # print(f"Key frame pose at theta {theta}: Position = {position}, Rotation = {rotation_matrix}")

            return key_frame_poses_in_world
        
        # define center and radius
        radius = 0.1
        p0 = [0.45, 0.0, 0.4]
        p_base = [0.0, 0.0, 0.0]
        R0 = RotationMatrix(np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]).T)
        X_WCenter = RigidTransform(R0, p0)

        num_key_frames = 10
        """
        you may use different thetas as long as your trajectory starts
        from the Start Frame above and your rotation is positive
        in the world frame about +z axis
        thetas = np.linspace(0, 2*np.pi, num_key_frames)
        """
        thetas = np.linspace(0, 2 * np.pi, num_key_frames)

        key_frame_poses = compose_circular_key_frames(thetas, X_WCenter, radius)
        # print("key_frame_poses: ", key_frame_poses)
        # print("length of key_frame_poses: ", len(key_frame_poses))

        # X_WGinit = get_X_WG(diagram,context=context)
        # print("X_WGinit: ", X_WGinit)
        X_WGinit = X_WG

        total_time = 20.0
        key_frame_poses = [X_WGinit] + compose_circular_key_frames(thetas, X_WCenter, radius)

        print(meshcat)
        if meshcat is not None:
            for i, frame in enumerate(key_frame_poses):
                AddMeshcatTriad(meshcat, X_PT=frame, path="frame"+str(i))

        times = np.linspace(start_time_s, start_time_s + total_time, num_key_frames + 1)

        return PiecewisePoseWithTimingInformation(
            trajectory=PiecewisePose.MakeLinear(times, key_frame_poses),
            start_time_s=start_time_s
        )

    # def traj_linear_move_to_bowl_0(self, context: Context, move_time=10) -> PiecewisePose:
    #     start_time = context.get_time()
    #     context_plant = self._iiwa_plant.CreateDefaultContext() # TODO: This is a problem spot.
    #     X_WGinit = self._iiwa_plant.EvalBodyPoseInWorld(context_plant, self._iiwa_plant.GetBodyByName("body"))
    #     int_1 = hw.RigidTransform(R=hw.RotationMatrix(), p=np.array([0, 1, 1]))
    #     int_2 = hw.RigidTransform(R=hw.RotationMatrix(), p=np.array([-4, 1, 1]))
    #     goal_pose = hw.RigidTransform(R=hw.RotationMatrix(), p=np.array(env.bowl_0) + np.array([0.25, 0, 0.25]))
    #     poses = hw.PiecewisePose.MakeLinear(
    #         [start_time, start_time + move_time / 4, start_time + 3 * move_time / 4, start_time + move_time],
    #         [X_WGinit, int_1, int_2, goal_pose]
    #     )
    #     return poses

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


