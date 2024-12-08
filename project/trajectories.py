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
    DifferentialInverseKinematicsParameters,
    InputPortIndex,
    LeafSystem,
    MultibodyPlant,
    PathParameterizedTrajectory,
    PiecewisePose,
    PortSwitch,
    RigidTransform,
    ImageDepth32F,
    ImageRgba8U,
    PointCloud,
    State,
)
import pizza_state as ps
import time

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
    EXECUTE_PLANNED_TRAJECTORY = 4
    FINISHED = 5
    FIX_BASE = 6
    EVAL_PIZZA_STATE = 7
    CLOSE_GRIPPER = 8
    OPEN_GRIPPER = 9
    LOCK_ARM = 10


class PizzaPlanner(LeafSystem):
    def __init__(self, 
        num_joint_positions: int,
        initial_delay_s: int,
        controller_plant: MultibodyPlant,
        diff_ik_controller: Diagram
    ):
        super().__init__()
        self._is_finished = False
        self._iiwa_plant = controller_plant
        self._gripper_frame = self._iiwa_plant.GetFrameByName("body")
        gripper_body = self._iiwa_plant.GetBodyByName("body")
        self._gripper_body_index = gripper_body.index()
        self._num_joint_positions = num_joint_positions

        plant_context = self._iiwa_plant.CreateDefaultContext()

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

        # Initialize diff_ik_params
        self._initial_diff_ik_params = diff_ik_controller.GetSubsystemByName("integrator").get_parameters()
        
        # diff ik control params
        self._joint_velocity_limits_idx = int(
            self.DeclareDiscreteState(2 * num_joint_positions)
        )
        self._ee_velocity_limits_idx = int(
            self.DeclareDiscreteState(6)
        )
        self._nominal_joint_position_idx = int(
            self.DeclareDiscreteState(num_joint_positions)
        )
        self._gripper_position_idx = int(
            self.DeclareDiscreteState(1)
        )

        # self._joint_centering_gains_idx = int(
        #     self.DeclareDiscreteState(num_joint_positions)
        # )

        self._camera_0_depth_image_idx = int(
            self.DeclareAbstractState(AbstractValue.Make(ImageDepth32F()))
        )
        self._camera_0_rgb_image_idx = int(
            self.DeclareAbstractState(AbstractValue.Make(ImageRgba8U()))
        )
        self._camera_0_point_cloud_idx = int(
            self.DeclareAbstractState(AbstractValue.Make(PointCloud()))
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

        self._desired_pose_idx = self.DeclareAbstractState(
            AbstractValue.Make(default_gripper_pose)
        )

        # ------------ CAMERA INPUT PORTS ------------ #
        self._camera_depth_input_port = self.DeclareAbstractInputPort(
            "camera_0.depth_image",
            AbstractValue.Make(ImageDepth32F())
        )
        self._camera_rgb_input_port = self.DeclareAbstractInputPort(
            "camera_0.rgb_image",
            AbstractValue.Make(ImageRgba8U())
        )
        self._camera_ptcloud_input_port = self.DeclareAbstractInputPort(
            "camera_0.point_cloud",
            AbstractValue.Make(PointCloud())
        )

        # OUTPUT PORTS (keep the order)
        self.DeclareAbstractOutputPort(
            "desired_pose",
            alloc=lambda: AbstractValue.Make(RigidTransform()),
            calc=self._set_desired_pose
        )

        # TODO: remove this output port if we don't need?
        # self.DeclareVectorOutputPort(
        #     "current_iiwa_positions", num_joint_positions, self._get_current_iiwa_positions
        # )

        self.DeclareVectorOutputPort(
            "joint_velocity_limits",
            2 * num_joint_positions,
            self._calc_joint_velocity_limits,
        )
        self.DeclareVectorOutputPort(
            "ee_velocity_limits",
            6,
            self._calc_ee_velocity_limits,
        )
        self.DeclareVectorOutputPort(
            "nominal_joint_position",
            num_joint_positions,
            self._calc_nominal_joint_position,
        )
        self.DeclareVectorOutputPort(
            "gripper_position",
            1,
            self._set_gripper_position
        )

        # self.DeclareVectorOutputPort(
        #     "joint_centering_gains",
        #     num_joint_positions,
        #     self._calc_joint_centering_gains,
        # )

        self.DeclareInitializationDiscreteUpdateEvent(self._initialize_discrete_state)
        self.DeclarePerStepUnrestrictedUpdateEvent(self._run_fsm_logic)

    def _get_current_iiwa_positions(self, context: Context, output: BasicVector) -> None:
        positions = context.get_discrete_state(self._current_iiwa_positions_idx).get_value()
        output.set_value(positions)

    def _solve_gripper_pose(self, context: Context) -> RigidTransform:
        body_poses = self._body_poses_input_port.Eval(context)
        return body_poses[self._gripper_body_index]

    def _set_desired_pose(self, context: Context, output: AbstractValue) -> None:
        desired_pose = context.get_abstract_state(self._desired_pose_idx).get_value()
        output.set_value(desired_pose)

    def _set_gripper_position(self, context: Context, output: BasicVector) -> None:
        output.set_value(context.get_discrete_state(self._gripper_position_idx).get_value())

    def _calc_joint_velocity_limits(self, context: Context, output: BasicVector) -> None:
        output.set_value(context.get_discrete_state(self._joint_velocity_limits_idx).get_value())

    def _calc_ee_velocity_limits(self, context: Context, output: BasicVector) -> None:
        output.set_value(context.get_discrete_state(self._ee_velocity_limits_idx).get_value())
    
    def _calc_nominal_joint_position(self, context: Context, output: BasicVector) -> None:
        output.set_value(context.get_discrete_state(self._nominal_joint_position_idx).get_value())

    def _grab_depth_image(self, context: Context) -> ImageDepth32F:
        return context.get_abstract_state(self._camera_0_depth_image_idx).get_value()
    
    def _grab_rgb_image(self, context: Context) -> ImageRgba8U:
        return context.get_abstract_state(self._camera_0_rgb_image_idx).get_value()

    def _grab_point_cloud(self, context: Context) -> PointCloud:
        point_cloud = self._camera_ptcloud_input_port.Eval(context)
        context.get_abstract_state(self._camera_0_point_cloud_idx).set_value(point_cloud)
        return point_cloud

    # def _calc_joint_centering_gains(self, context: Context, output: BasicVector) -> None:
    #     output.set_value(context.get_discrete_state(self._joint_centering_gains_idx).get_value())
  
    # TODO: probably dead for now
    # def _calc_diff_ik_reset(self, context: Context, output: AbstractValue) -> None:
    #     # credit: used with permission from Nicholas.
    #     """Logic for deciding when to reset the differential IK controller state."""
    #     state = context.get_abstract_state(int(self._fsm_state_idx)).get_value()
    #     if state == PizzaRobotState.PLAN_IIWA_PAINTER:
    #         # Need to reset before executing a trajectory.
    #         output.set_value(True)
    #     else:
    #         output.set_value(False)

    # init function
    def _initialize_discrete_state(self, context: Context, discrete_values: DiscreteValues) -> None:
        discrete_values.get_mutable_vector(self._current_iiwa_positions_idx).SetFromVector(
            self._iiwa_state_estimated_input_port.Eval(context)[:10]
        )

        lower_limits, upper_limits = self._initial_diff_ik_params.get_joint_velocity_limits()
        joint_velocity_limits = np.concatenate((lower_limits, upper_limits))
        discrete_values.get_mutable_vector(self._joint_velocity_limits_idx).set_value(joint_velocity_limits)

        lower_limits, upper_limits = self._initial_diff_ik_params.get_end_effector_translational_velocity_limits()
        ee_vel_limits = np.concatenate((lower_limits, upper_limits))
        discrete_values.get_mutable_vector(self._ee_velocity_limits_idx).set_value(ee_vel_limits)

        discrete_values.get_mutable_vector(self._nominal_joint_position_idx).set_value(
            self._initial_diff_ik_params.get_nominal_joint_position()
        )

        discrete_values.get_mutable_vector(self._gripper_position_idx).set_value([0.107])

        # discrete_values.get_mutable_vector(self._joint_centering_gains_idx).set_value(
        #     self._initial_diff_ik_params.get_joint_centering_gain()
        # )

        # discrete_values.get_mutable_value(self._camera_0_depth_image_idx).set_value(
        #     self._camera_depth_input_port.Eval(context)
        # )
        # discrete_values.get_mutable_value(self._camera_0_rgb_image_idx).set_value(
        #     self._camera_rgb_input_port.Eval(context)
        # )
        # discrete_values.get_mutable_value(self._camera_0_point_cloud_idx).set_value(
        #     self._camera_ptcloud_input_port.Eval(context)
        # )
        

    ###------------ STATE MACHINE LOGIC -----------###
    def _run_fsm_logic(self, context: Context, state: State) -> None:
        current_time = context.get_time()
        timing_information: PizzaBotTimingInformation = (
            context.get_mutable_abstract_state(self._timing_information_idx).get_value()
        )
        mutable_fsm_state = state.get_mutable_abstract_state(self._fsm_state_idx)
        fsm_state_value: PizzaRobotState = context.get_abstract_state(self._fsm_state_idx).get_value()

        def transition_to_state(next_state: PizzaRobotState):
            print(f"Transitioning to {next_state} FSM state.")
            mutable_fsm_state.set_value(next_state)

        if self._is_finished:
            return

        # ----------------- START ----------------- #
        if fsm_state_value == PizzaRobotState.START:
            print("Current state: START")
            q_current = self._iiwa_state_estimated_input_port.Eval(context)[:10]
            state.get_mutable_discrete_state().set_value(self._current_iiwa_positions_idx, q_current)

            # keep the arm on the ground for now.
            joint_vels = self.calc_limits_fix_base_position(context, [0, 0, 1])
            state.get_mutable_discrete_state().set_value(self._joint_velocity_limits_idx, joint_vels)

            transition_to_state(PizzaRobotState.CLOSE_GRIPPER)
        
        # ----------------- PLAN IIWA_PAINTER ----------------- #
        elif fsm_state_value == PizzaRobotState.PLAN_IIWA_PAINTER:
            gripper_pose = self._solve_gripper_pose(context)
            pose_traj = self._create_painter_traj(
                gripper_pose,
                start_time_s=timing_information.start_iiwa_painter,
                context=context
            )
            state.get_mutable_abstract_state(self._pose_trajectory_idx).set_value(pose_traj)

            mutable_fsm_state.set_value(PizzaRobotState.FIX_BASE)
            print("Transitioning to EXECUTE_IIWA_PAINTER FSM state.")

        # ----------------- FIX BASE ----------------- #
        elif fsm_state_value == PizzaRobotState.FIX_BASE:
            new_joint_lims = self.calc_limits_fix_base_position(context, [0, 0, 1])
            state.get_mutable_discrete_state().set_value(self._joint_velocity_limits_idx, new_joint_lims)

            transition_to_state(PizzaRobotState.EXECUTE_PLANNED_TRAJECTORY)
        

        # ----------------- PLAN TRAJECTORY_0 ----------------- #
        elif fsm_state_value == PizzaRobotState.PLAN_TRAJ_0:
            pose_traj = self.traj_linear_move_to_bowl_0(context)
            state.get_mutable_abstract_state(self._pose_trajectory_idx).set_value(
                PiecewisePoseWithTimingInformation(trajectory=pose_traj, start_time_s=current_time)
            )
            mutable_fsm_state.set_value(PizzaRobotState.EXECUTE_PLANNED_TRAJECTORY)
            print("Transitioning to EXECUTE_PLANNED_TRAJECTORY state.")
        

        # ----------------- EVALUATE PIZZA STATE ----------------- #
        elif fsm_state_value == PizzaRobotState.EVAL_PIZZA_STATE:
            # print("Evaluating pizza state.")
            point_cloud = self._grab_point_cloud(context)
            print(point_cloud)

            # test code
            points = point_cloud.xyzs().T
            print(points)
            colors = point_cloud.rgbs().T.reshape(-1, 3)/255.0
            area_of_pizza = ps.calculate_pizza_area(points, colors)
            print(area_of_pizza)

            mutable_fsm_state.set_value(PizzaRobotState.FINISHED)
            print("Transitioning to FINISHED state.")


        # ----------------- CLOSE GRIPPER ----------------- #
        elif fsm_state_value == PizzaRobotState.CLOSE_GRIPPER:
            start_gripper_position = context.get_discrete_state(self._gripper_position_idx).get_value()[0]
            
            # runs this once, 
            if not hasattr(self, '_gripper_traj'):
                self._gripper_traj = self._close_gripper_traj(context,start_gripper_position,current_time)
            gripper_traj = self._gripper_traj

            gripper_position_desired = gripper_traj.value(current_time)
            state.get_mutable_discrete_state().get_mutable_vector(self._gripper_position_idx).set_value(gripper_position_desired)
            
            if current_time >= gripper_traj.end_time():
                del(self._gripper_traj)
                transition_to_state(PizzaRobotState.OPEN_GRIPPER)

        # ----------------- OPEN GRIPPER ----------------- #
        elif fsm_state_value == PizzaRobotState.OPEN_GRIPPER:
            start_gripper_position = context.get_discrete_state(self._gripper_position_idx).get_value()[0]
            if not hasattr(self, '_gripper_traj'):
                self._gripper_traj = self._open_gripper_traj(
                    context,
                    start_gripper_position,
                    start_time_s=current_time) # start immediately on state transition
            gripper_traj = self._gripper_traj

            gripper_position_desired = gripper_traj.value(current_time)
            state.get_mutable_discrete_state().get_mutable_vector(self._gripper_position_idx).set_value(gripper_position_desired)
            
            if current_time >= gripper_traj.end_time():
                del(self._gripper_traj)
                transition_to_state(PizzaRobotState.FINISHED)

        
        
        # ----------------- EXECUTE PLANNED TRAJECTORY ----------------- #
        elif fsm_state_value == PizzaRobotState.EXECUTE_PLANNED_TRAJECTORY:

            pose_traj = context.get_abstract_state(self._pose_trajectory_idx).get_value()

            if current_time >= timing_information.start_iiwa_painter:
                desired_pose = pose_traj.trajectory.GetPose(current_time - timing_information.start_iiwa_painter)
                state.get_mutable_abstract_state(self._desired_pose_idx).set_value(desired_pose)
            
            if current_time >= pose_traj.start_time_s + pose_traj.trajectory.end_time():
                print("Transitioning to FINISHED state.")
                mutable_fsm_state.set_value(PizzaRobotState.FINISHED)


        # ----------------- FINISHED ----------------- #
        elif fsm_state_value == PizzaRobotState.FINISHED:
            self._is_finished = True
            print("Task is finished.")


    def calc_limits_fix_base_position(self, context: Context, xyz_fixed):
        tolerance = 1e-3

        # Get mutable joint velocity limits from discrete state
        joint_velocity_limits = context.get_mutable_discrete_state(self._joint_velocity_limits_idx).get_value()

        # Initialize new joint velocity limits
        new_joint_velocity_limits = np.copy(joint_velocity_limits)

        for i, axis_fixed in enumerate(xyz_fixed):
            if axis_fixed:
                # If fixed, set velocity limits to zero
                new_joint_velocity_limits[i] = 0.0
                new_joint_velocity_limits[i + self._num_joint_positions] = 0.0
            else:
                # If free, keep the nominal velocity limits
                init_vels = np.array(self._initial_diff_ik_params.get_joint_velocity_limits()).flatten()
                new_joint_velocity_limits[i] = init_vels[i]
                new_joint_velocity_limits[i + self._num_joint_positions] = init_vels[i + self._num_joint_positions]

        return new_joint_velocity_limits


    def set_arm_locked(self, context: Context, locked):
        joint_velocity_limits = context.get_mutable_discrete_state(self._joint_velocity_limits_idx).get_value()
        new_joint_velocity_limits = np.copy(joint_velocity_limits)
        init_vels = np.array(self._initial_diff_ik_params.get_joint_velocity_limits()).flatten()

        new_joint_velocity_limits[3:] = 0.0 if locked else init_vels[3:]
        
        return new_joint_velocity_limits
    

    def move_base_to_location(self, context: Context):
        pass


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

    def make_traj_move_arm(
        self,
        context: Context,
        start_pose,
        end_pose,
        start_time=None,
        move_time=10,
    ) -> PiecewisePose:
        if not start_time:
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

        # print(meshcat)
        if meshcat is not None:
            for i, frame in enumerate(key_frame_poses):
                AddMeshcatTriad(meshcat, X_PT=frame, path="frame"+str(i))

        times = np.linspace(start_time_s, start_time_s + total_time, num_key_frames + 1)

        return PiecewisePoseWithTimingInformation(
            trajectory=PiecewisePose.MakeLinear(times, key_frame_poses),
            start_time_s=start_time_s
        )

    def _close_gripper_traj(self, context: Context, start_gripper_position, start_time_s: int) -> PiecewisePolynomial:
        # Create gripper trajectory.
        end_gripper_position = 0.002
        move_time_s = 0.5

        gripper_t_lst = start_time_s + np.array([0.0, move_time_s])
        gripper_knots = np.array([start_gripper_position, end_gripper_position]).reshape(1,2)
        g_traj = PiecewisePolynomial.FirstOrderHold(gripper_t_lst, gripper_knots)

        return g_traj

    def _open_gripper_traj(self, context, start_gripper_position, start_time_s: int) -> PiecewisePolynomial:
        # Create gripper trajectory.
        end_gripper_position = 0.1
        move_time_s = 0.5

        gripper_t_lst = start_time_s + np.array([0.0, move_time_s])
        gripper_knots = np.array([start_gripper_position, end_gripper_position]).reshape(1,2)
        g_traj = PiecewisePolynomial.FirstOrderHold(gripper_t_lst, gripper_knots)

        return g_traj
