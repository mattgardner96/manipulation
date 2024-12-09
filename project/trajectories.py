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
    InverseKinematics,
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
    RollPitchYaw,
)
import pizza_state as ps
import time

from manipulation.meshcat_utils import AddMeshcatTriad
from inverse_kinematics import solve_global_inverse_kinematics

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
    EXECUTE_PLANNED_TRAJECTORY = 1
    FINISHED = 2
    EVAL_PIZZA_STATE = 3
    MOVE_TO_BOWL_0 = 4
    PUT_BACK_BOWL_0 = 5
    MOVE_TO_BOWL_1 = 6
    LIFT_BOWL_0 = 7
    LIFT_BOWL_1 = 8
    TEST_IK_MOVE = 9
    MOVE_TO_BREADPAN_QUEUE_SHAKE = 10
    SHIMMY_SHAKE = 11
    EXECUTE_SHIMMY_SHAKE = 12


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
        self.count = 0
        self._eval_required = False

        # Table 0 workstation = (-1, -0.25, 1.5) 
        # Table 1 workstation placement = (-2.5, 1, 1.5)
        # Oven workstation = (0, 1.5, 1)
        # Delivery workstation = (-0.5, 2, 1)
        BOWL_APPROACH_OFFSET = np.array([0.148,0.0,0.1])
        BREADPAN_APPROACH_OFFSET = np.array([0,0.5,0.4])
        self.workstation_poses = {
            "table_0": RigidTransform(
                RotationMatrix.MakeZRotation(np.pi),
                [-1,0.25,1]
            ),
            "table_1": RigidTransform(RotationMatrix.MakeZRotation(np.pi),[-2.5,1,1]),
            "oven": RigidTransform(RotationMatrix.MakeZRotation(0),[0,1.5,1]),
            "delivery": RigidTransform(RotationMatrix.MakeZRotation(np.pi/2),[-0.5,1.5,1]),
            "bowl_0": RigidTransform(
                RotationMatrix(RollPitchYaw(0,-np.pi/2,np.pi/2)),
                np.array(env.bowl_0)+BOWL_APPROACH_OFFSET
            ),
            "bowl_1": RigidTransform(
                RotationMatrix(RollPitchYaw(0,-np.pi/2,np.pi/2)),
                np.array(env.bowl_1)+BOWL_APPROACH_OFFSET
            ),
            "bowl_2": RigidTransform(
                RotationMatrix(RollPitchYaw(0,-np.pi/2,np.pi/2)),
                np.array(env.bowl_0)+BOWL_APPROACH_OFFSET
            ),
            # proximity move to approach any bowl. relative
            "bowl_lift_relative": RigidTransform(
                RotationMatrix(),
                p=[0.2,0,0]
            ),
            # proximity move to approach any bowl. relative
            "bowl_lower_relative": RigidTransform(
                RotationMatrix(),
                p=[-0.2,0,0]
            ),
            # absolute pose
            "breadpan_queue_shake": RigidTransform(
                RotationMatrix(RollPitchYaw(-np.pi/2, -np.pi/4, 3*np.pi/2)),
                p=env.pan_position+BREADPAN_APPROACH_OFFSET
            )
        }

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
        self._iiwa_positions_idx = int(
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

        # ----------------- INPUT PORTS ----------------- #
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

        self._control_mode_idx = self.DeclareAbstractState(
            AbstractValue.Make(InputPortIndex(1))
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

        # ----------------- OUTPUT PORTS ----------------- #
        # (keep the order)
        self.DeclareAbstractOutputPort(
            "desired_pose",
            alloc=lambda: AbstractValue.Make(RigidTransform()),
            calc=self._set_desired_pose
        )

        # remove this output port if we don't need?
        self.DeclareVectorOutputPort(
            "iiwa_positions", num_joint_positions, self._set_iiwa_positions
        )

        self._joint_output_port = self.DeclareVectorOutputPort(
            "joint_velocity_limits",
            2 * num_joint_positions,
            self._calc_joint_velocity_limits
        )
        self.DeclareVectorOutputPort(
            "ee_velocity_limits",
            6,
            self._calc_ee_velocity_limits
        )
        self.DeclareVectorOutputPort(
            "nominal_joint_position",
            num_joint_positions,
            self._calc_nominal_joint_position
        )
        self.DeclareVectorOutputPort(
            "gripper_position",
            1,
            self._set_gripper_position
        )

        self.DeclareAbstractOutputPort(
            "control_mode", 
            lambda: AbstractValue.Make(InputPortIndex(1)),
            self._calc_control_mode
        )

        self.DeclareAbstractOutputPort(
            "reset_diff_ik",
            lambda: AbstractValue.Make(False),
            self._get_reset_diff_ik
        )


        # self.DeclareVectorOutputPort(
        #     "joint_centering_gains",
        #     num_joint_positions,
        #     self._calc_joint_centering_gains,
        # )

        self.DeclareInitializationDiscreteUpdateEvent(self._initialize_discrete_state)
        self.DeclarePerStepUnrestrictedUpdateEvent(self._run_fsm_logic)

    def _calc_control_mode(self, context: Context, output: BasicVector) -> None:
        """
        Outputs 0 for joint trajectory following or 1 for task space differential inverse kinematics.
        """
        mode = context.get_abstract_state(self._control_mode_idx).get_value()
        if not isinstance(mode, InputPortIndex):
            print("how this happen")
            mode = InputPortIndex(1)  # Default to joint trajectory following
        output.set_value(mode)

    def _get_reset_diff_ik(self, context: Context, output: AbstractValue) -> None:
        mode = context.get_abstract_state(self._control_mode_idx).get_value()
        if mode == InputPortIndex(2):
            output.set_value(False)
        else:
            output.set_value(True)

    def _set_iiwa_positions(self, context: Context, output: BasicVector) -> None:
        positions = context.get_discrete_state(self._iiwa_positions_idx).get_value()
        output.set_value(positions)

    def _solve_gripper_pose(self, context: Context) -> RigidTransform:
        body_poses = self._body_poses_input_port.Eval(context)
        return body_poses[self._gripper_body_index]

    def _set_desired_pose(self, context: Context, output: AbstractValue) -> None:
        control_mode = context.get_abstract_state(self._control_mode_idx).get_value()
        if control_mode == InputPortIndex(1):
            desired_pose = self._solve_gripper_pose(context) # desired equal to current pose
            output.set_value(self._solve_gripper_pose(context))
        else:
            desired_pose = context.get_abstract_state(self._desired_pose_idx).get_value()
            output.set_value(desired_pose)

    def _set_gripper_position(self, context: Context, output: BasicVector) -> None:
        output.set_value(context.get_discrete_state(self._gripper_position_idx).get_value())

    def _calc_joint_velocity_limits(self, context: Context, output: BasicVector) -> None:
        output.set_value(context.get_discrete_state(self._joint_velocity_limits_idx).get_value())

    def _calc_ee_velocity_limits(self, context: Context, output: BasicVector) -> None:
        output.set_value(context.get_discrete_state(self._ee_velocity_limits_idx).get_value())
    
    def _calc_nominal_joint_position(self, context: Context, output: BasicVector) -> None:
        if context.get_abstract_state(self._control_mode_idx).get_value() == InputPortIndex(1):
            val = self._iiwa_state_estimated_input_port.Eval(context)[:10]
            output.set_value(val)
            context.get_mutable_discrete_state().set_value(self._nominal_joint_position_idx, val)
        else:
            output.set_value(context.get_discrete_state(self._nominal_joint_position_idx).get_value())

    def _grab_depth_image(self, context: Context) -> ImageDepth32F:
        return context.get_abstract_state(self._camera_0_depth_image_idx).get_value()
    
    def _grab_rgb_image(self, context: Context) -> ImageRgba8U:
        return context.get_abstract_state(self._camera_0_rgb_image_idx).get_value()

    def _grab_point_cloud(self, context: Context) -> PointCloud:
        point_cloud = self._camera_ptcloud_input_port.Eval(context)
        context.get_abstract_state(self._camera_0_point_cloud_idx).set_value(point_cloud)
        return point_cloud

        
    def _initialize_discrete_state(self, context: Context, discrete_values: DiscreteValues) -> None:
        
        discrete_values.get_mutable_vector(self._iiwa_positions_idx).SetFromVector(
            self._iiwa_state_estimated_input_port.Eval(context)[:10]
        )
        # print(discrete_values.get_mutable_vector(self._iiwa_positions_idx).get_value())

        lower_limits, upper_limits = self._initial_diff_ik_params.get_joint_velocity_limits()
        joint_vel_limits = np.concatenate((lower_limits, upper_limits))
        discrete_values.get_mutable_vector(self._joint_velocity_limits_idx).set_value(joint_vel_limits)

        lower_limits, upper_limits = self._initial_diff_ik_params.get_end_effector_translational_velocity_limits()
        ee_vel_limits = np.concatenate((lower_limits, upper_limits))
        discrete_values.get_mutable_vector(self._ee_velocity_limits_idx).set_value(ee_vel_limits)

        discrete_values.get_mutable_vector(self._nominal_joint_position_idx).set_value(
            self._initial_diff_ik_params.get_nominal_joint_position()
        )

        discrete_values.get_mutable_vector(self._gripper_position_idx).set_value([0.107])

        context.get_mutable_abstract_state(self._control_mode_idx).set_value(InputPortIndex(2)) # diff ik

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


    ###------------------------------------------------------------------STATE MACHINE PLANNER-------------------------------------------------------------------###    

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

        def set_control_mode(mode: str):
            if mode == "joint":
                state.get_mutable_abstract_state(self._control_mode_idx).set_value(InputPortIndex(1))
                context.get_mutable_abstract_state(self._control_mode_idx).set_value(InputPortIndex(1))
            elif mode == "diff_ik":
                state.get_mutable_abstract_state(self._control_mode_idx).set_value(InputPortIndex(2))
                context.get_mutable_abstract_state(self._control_mode_idx).set_value(InputPortIndex(2))

        if self._is_finished:
            return

        # ----------------- START ----------------- #
        if fsm_state_value == PizzaRobotState.START:
            
            set_control_mode("joint")

            #print("Current state: START")
            q_current = self._iiwa_state_estimated_input_port.Eval(context)[:10]
            state.get_mutable_discrete_state().set_value(self._iiwa_positions_idx, q_current)

            # state.get_mutable_abstract_state(self._control_mode_idx).set_value(InputPortIndex(1)) # set to diffIK mode

            # keep the robot on the ground for now.
            new_joint_lims = self.calc_limits_fix_base_position(context, xyz_fixed=[0, 0, 1])
            state.get_mutable_discrete_state().set_value(self._joint_velocity_limits_idx, new_joint_lims)

            transition_to_state(PizzaRobotState.MOVE_TO_BOWL_0)

        # ----------------- EVALUATE PIZZA STATE ----------------- #
        elif fsm_state_value == PizzaRobotState.EVAL_PIZZA_STATE:
            PIZZA_COVERED_THRESHOLD = 0.7
            
            if self._eval_required:
                # print("Evaluating pizza state.")
                point_cloud = self._grab_point_cloud(context)

                # test code
                points = point_cloud.xyzs().T
                print(points)
                colors = point_cloud.rgbs().T.reshape(-1, 3)/255.0
                area_of_pizza = ps.calculate_pizza_area(points, colors)
                print(area_of_pizza)
                self._eval_required = False

            # TODO: real logic here

            if area_of_pizza >= PIZZA_COVERED_THRESHOLD:
                transition_to_state(PizzaRobotState.PUT_BACK_BOWL_0)
                self.count = 0
            else:
                transition_to_state(PizzaRobotState.MOVE_TO_BREADPAN_QUEUE_SHAKE)
                self.count = 0
            


        # ----------------- MOVE TO BOWL 0 ----------------- #
        elif fsm_state_value == PizzaRobotState.MOVE_TO_BOWL_0:

            set_control_mode("joint")

            curr_pose = self._body_poses_input_port.Eval(context)[self._gripper_body_index]
            goal_pose = self.workstation_poses["bowl_0"]

            if self.count == 0:
                if self._ik_move_to_posn(context, state, goal_pose, [0,0,1]) == False:
                    return

            if self.count == 1:
                # close gripper
                if self.gripper_action(context, current_time, state, "close") == True:
                    transition_to_state(PizzaRobotState.LIFT_BOWL_0)
        

        # ----------------- LIFT BOWL 0 ----------------- #
        elif fsm_state_value == PizzaRobotState.LIFT_BOWL_0:
            set_control_mode("diff_ik")

            # # lock z, unlock xy
            # new_joint_lims = self.calc_limits_fix_base_position(context, xyz_fixed=[0,0,1])
            # state.get_mutable_discrete_state().set_value(self._joint_velocity_limits_idx, new_joint_lims)

            curr_pose = self._body_poses_input_port.Eval(context)[self._gripper_body_index]
            
            goal_pose = curr_pose @ self.workstation_poses["bowl_lift_relative"]

            if self.count == 0:
                if self._diff_ik_move_to_posn(context, state, goal_pose,[0,0,1]) == False:
                    return
            
            self.count = 0
            transition_to_state(PizzaRobotState.MOVE_TO_BREADPAN_QUEUE_SHAKE)



        # ----------------- PROX MOVE TO BOWL 1 ----------------- #
        elif fsm_state_value == PizzaRobotState.MOVE_TO_BOWL_1:

            set_control_mode("joint")

            # print(self.GetOutputPort("control_mode").Eval(context))

            curr_pose = self._body_poses_input_port.Eval(context)[self._gripper_body_index]
            goal_pose = self.workstation_poses["bowl_1"]

            if self.count == 0:
                if self._ik_move_to_posn(context, state, goal_pose, [0,0,1] ) == False:
                    return
            
            if self.count == 1:
                # close gripper
                if self.gripper_action(context, current_time, state, "close") == True:
                    transition_to_state(PizzaRobotState.LIFT_BOWL_1)


        # ----------------- LIFT BOWL 1 ----------------- #
        elif fsm_state_value == PizzaRobotState.LIFT_BOWL_1:

            set_control_mode("diff_ik")
            
            # # lock z, unlock xy
            # new_joint_lims = self.calc_limits_fix_base_position(context, xyz_fixed=[0,0,1])
            # state.get_mutable_discrete_state().set_value(self._joint_velocity_limits_idx, new_joint_lims)

            curr_pose = self._body_poses_input_port.Eval(context)[self._gripper_body_index]
            
            goal_pose = curr_pose @ self.workstation_poses["bowl_lift_relative"]

            if self.count == 0:
                if self._diff_ik_move_to_posn(context, state, goal_pose,[0,0,1]) == False:
                    return
            
            transition_to_state(PizzaRobotState.MOVE_TO_BREADPAN_QUEUE_SHAKE)
            self.count = 0

        # ----------------- MOVE TO BREADPAN QUEUE SHAKE ----------------- #
        elif fsm_state_value == PizzaRobotState.MOVE_TO_BREADPAN_QUEUE_SHAKE:
            
            if self._eval_required:
                transition_to_state(PizzaRobotState.EVAL_PIZZA_STATE)
                return

            set_control_mode("joint")

            curr_pose = self._body_poses_input_port.Eval(context)[self._gripper_body_index]
            goal_pose = self.workstation_poses["breadpan_queue_shake"]

            if self.count == 0:
                if self._ik_move_to_posn(context, state, goal_pose, [0,0,1],move_time=4) == False:
                    return

            transition_to_state(PizzaRobotState.SHIMMY_SHAKE)


        # ----------------- SHIMMY SHAKE ----------------- #
        elif fsm_state_value == PizzaRobotState.SHIMMY_SHAKE:
            set_control_mode("diff_ik")

            # lock the base in x and z
            new_joint_lims = self.calc_limits_fix_base_position(context, xyz_fixed=[1,0,1])
            state.get_mutable_discrete_state().set_value(self._joint_velocity_limits_idx, new_joint_lims)

            curr_pose = self._body_poses_input_port.Eval(context)[self._gripper_body_index]

            # shimmy shake
            shimmy_traj = self._get_shimmy_traj(
                context,
                num_keyframes=20,
                amplitude=0.1,
                frequency=100
            )

            state.get_mutable_abstract_state(self._pose_trajectory_idx).set_value(shimmy_traj)
            # context.get_mutable_abstract_state(self._pose_trajectory_idx).set_value(shimmy_traj)

            transition_to_state(PizzaRobotState.EXECUTE_SHIMMY_SHAKE)
        

        # ----------------- EXECUTE PLANNED TRAJECTORY ----------------- #
        elif fsm_state_value == PizzaRobotState.EXECUTE_SHIMMY_SHAKE:
            
            set_control_mode("diff_ik")

            pose_traj = state.get_abstract_state(self._pose_trajectory_idx).get_value()

            if current_time >= pose_traj.start_time_s:
                elapsed_time = current_time - pose_traj.start_time_s
                desired_pose = pose_traj.trajectory.GetPose(elapsed_time)
                state.get_mutable_abstract_state(self._desired_pose_idx).set_value(desired_pose)
            
            if current_time >= pose_traj.start_time_s + pose_traj.trajectory.end_time():
                self._eval_required = True
                transition_to_state(PizzaRobotState.MOVE_TO_BREADPAN_QUEUE_SHAKE)


        # ----------------- PUT BACK BOWL 1 ----------------- #
        elif fsm_state_value == PizzaRobotState.PUT_BACK_BOWL_0:

            set_control_mode("joint")

            curr_pose = self._body_poses_input_port.Eval(context)[self._gripper_body_index]
            goal_pose = self.workstation_poses["bowl_0"] @ self.workstation_poses["bowl_lift_relative"]

            if self.count == 0:
                if self._ik_move_to_posn(context, state, goal_pose, [0,0,1] ) == False:
                    return
            
            if self.count == 1:
                # open gripper
                if self.gripper_action(context, current_time, state, "open") == True:
                    transition_to_state(PizzaRobotState.EVAL_PIZZA_STATE)


        

        # ----------------- EVALUATE PIZZA STATE ----------------- #
        elif fsm_state_value == PizzaRobotState.EVAL_PIZZA_STATE:
            # print("Evaluating pizza state.")
            point_cloud = self._grab_point_cloud(context)

            # test code
            points = point_cloud.xyzs().T
            print(points)
            colors = point_cloud.rgbs().T.reshape(-1, 3)/255.0
            area_of_pizza = ps.calculate_pizza_area(points, colors)
            print(area_of_pizza)

            mutable_fsm_state.set_value(PizzaRobotState.FINISHED)
            print("Transitioning to FINISHED state.")

        # ----------------- FINISHED ----------------- #
        elif fsm_state_value == PizzaRobotState.FINISHED:
            self._is_finished = True
            print("Task is finished.")



### --------------HELPER FUNCTIONS------------------- ###

    def calc_limits_fix_base_position(self, context: Context, xyz_fixed):
        """Function to fix the movable base in set position"""

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
        """Function to fix arm joints (and therefore only move base) ---> currently not working"""

        joint_velocity_limits = context.get_mutable_discrete_state(self._joint_velocity_limits_idx).get_value()
        new_joint_velocity_limits = np.copy(joint_velocity_limits)
        init_vels = np.array(self._initial_diff_ik_params.get_joint_velocity_limits()).flatten()

        joint_velocity_limits = context.get_mutable_discrete_state().get_value(self._joint_velocity_limits_idx)
        new_joint_velocity_limits = np.copy(joint_velocity_limits)
        print(f"joint lims: {joint_velocity_limits}")

        new_joint_velocity_limits[3:10] = 0.0 if locked else init_vels[3:10]
        new_joint_velocity_limits[13:] = 0.0 if locked else init_vels[13:]
        
        return new_joint_velocity_limits
    

    """Trajectory making for moving iiwa arm"""
    def make_traj_move_arm(
        self, context: Context, start_pose, goal_pose, intermediate_poses=None, start_time=None, move_time=5,
    ) -> PiecewisePose:
        '''
        context: planner Context object
        start_pose: RigidTransform for entry position
        goal_pose: RigidTransform for exit position
        intermediate_poses: list of RigidTransforms for intermediate points as desired
        start_time: time to start the trajectory (default is current time)
        move_time: elapsed time of trajectory
        '''
        if not start_time:
            start_time = context.get_time()

        X_WGinit = start_pose

        # handle intermediate poses
        times = [start_time]
        if intermediate_poses is not None:
            # linear interpolation between intermediate poses
            intermediate_times = np.linspace(start_time, start_time + move_time, len(intermediate_poses)+2)[1:-1]
            times.extend(intermediate_times)
        times.append(start_time + move_time)

        # handle intermediate poses
        moves = [X_WGinit]
        if intermediate_poses is not None:
            moves.extend(intermediate_poses)
        moves.append(goal_pose)

        poses = PiecewisePose.MakeLinear(
            times,
            moves
        )
        return poses

#example trajectory making from class -> useful background 
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
        # Create close gripper trajectory.
        end_gripper_position = 0.000
        move_time_s = 0.5

        gripper_t_lst = start_time_s + np.array([0.0, move_time_s])
        gripper_knots = np.array([start_gripper_position, end_gripper_position]).reshape(1,2)
        g_traj = PiecewisePolynomial.FirstOrderHold(gripper_t_lst, gripper_knots)

        return g_traj

    def _open_gripper_traj(self, context, start_gripper_position, start_time_s: int) -> PiecewisePolynomial:
        # Create open gripper trajectory.
        end_gripper_position = 0.1
        move_time_s = 0.5

        gripper_t_lst = start_time_s + np.array([0.0, move_time_s])
        gripper_knots = np.array([start_gripper_position, end_gripper_position]).reshape(1,2)
        g_traj = PiecewisePolynomial.FirstOrderHold(gripper_t_lst, gripper_knots)

        return g_traj


    def gripper_action(self,context,current_time, state, action):
        """ 
        Make the gripper execute trajectory to close and open the gripper.
        Precondition: self.count is equal to 1.
        Args:
            action: "close" or "open" 
        """
        start_gripper_position = context.get_discrete_state(self._gripper_position_idx).get_value()[0]

        #if trajectory does not exist, make it, else execute next time step
        if not hasattr(self, '_gripper_traj') and action == "close":
            self._gripper_traj = self._close_gripper_traj(context, start_gripper_position, current_time)
        
        if not hasattr(self, '_gripper_traj') and action == "open":
            self._gripper_traj = self._open_gripper_traj(context, start_gripper_position, current_time)
        
        gripper_traj = self._gripper_traj
    
        while current_time < gripper_traj.end_time():
            gripper_position_desired = gripper_traj.value(current_time)

            #Update Gripper position    
            state.get_mutable_discrete_state().get_mutable_vector(self._gripper_position_idx).set_value(gripper_position_desired)
            return False
            
        del(self._gripper_traj)
        self.count = 0
        return True

    def _diff_ik_move_to_posn(self, context: Context, state: State, goal_pose: RigidTransform, fix_base) -> bool:
        """ 
        plans and executes a linear trajectory to a goal pose
        usage within state machine logic: 
        if self._diff_ik_move_to_posn(context, state, goal_pose) == False:
            return
        else:
            transition_to_state(PizzaRobotState.FINISHED)
        """

        curr_pose = self._body_poses_input_port.Eval(context)[self._gripper_body_index]
        
        if not hasattr(self, '_pose_traj_move_to_bowl_0'):
            pose_traj = self.make_traj_move_arm(
                context,
                start_pose=curr_pose,
                goal_pose=goal_pose,
                start_time=context.get_time(),
                move_time=2.0
            )
            new_traj = PiecewisePoseWithTimingInformation(trajectory=pose_traj, start_time_s=context.get_time())
            state.get_mutable_abstract_state(self._pose_trajectory_idx).set_value(new_traj)
            self._pose_traj_move_to_bowl_0 = new_traj

        pose_traj = self._pose_traj_move_to_bowl_0

        if context.get_time() >= pose_traj.start_time_s:
            elapsed_time = context.get_time() - pose_traj.start_time_s
            desired_pose = pose_traj.trajectory.GetPose(elapsed_time)
            state.get_mutable_abstract_state(self._desired_pose_idx).set_value(desired_pose)
        
        if context.get_time() >= pose_traj.start_time_s + pose_traj.trajectory.end_time():
            del self._pose_traj_move_to_bowl_0
            self.count = 1
            return True
        else:
            return False


    def _ik_move_to_posn(self, context: Context, state: State, goal_pose: RigidTransform, fix_base, move_time=2.0) -> bool:
        
        if not hasattr(self, '_joint_traj'):
            # Compute the current and goal joint positions
            q_current = np.append(self._iiwa_state_estimated_input_port.Eval(context)[:10],np.zeros(2))
            q_goal = solve_global_inverse_kinematics(
                plant=self._iiwa_plant,
                X_G=goal_pose,
                initial_guess=q_current,
                position_tolerance=0.0,
                orientation_tolerance=0.0,
                gripper_frame_name=self._gripper_frame.name(),
                fix_base=fix_base
            )

            if q_goal is None:
                print("No IK solution found.")
                return False  # Movement cannot be performed

            # Create a joint trajectory from current to goal positions
            current_time = context.get_time()
            times = [current_time, current_time + move_time]
            knots = np.column_stack((q_current, q_goal))
            self._joint_traj = PiecewisePolynomial.FirstOrderHold(times, knots)
            self._joint_traj_start_time = current_time
            return False  # Trajectory created, movement not yet completed

        # Execute the trajectory
        elapsed_time = context.get_time() - self._joint_traj_start_time
        if elapsed_time <= self._joint_traj.end_time():
            q_desired = self._joint_traj.value(elapsed_time).flatten()
            state.get_mutable_discrete_state(self._iiwa_positions_idx).set_value(q_desired[:10])
            return False  # Movement in progress
        else:
            # Trajectory completed
            del self._joint_traj
            del self._joint_traj_start_time
            self.count = 1
            return True  # Movement completed
    

    def _get_shimmy_traj(self, context: Context, num_keyframes, amplitude, frequency):
        """make shaking movement on top of pizza to spread ingredients"""

        X_WG_init = self._body_poses_input_port.Eval(context)[self._gripper_body_index]
        y_start = X_WG_init.translation()[1]
        start_time_s = context.get_time()
        move_time = 5
       
        key_frame_poses_in_world = []  
        amplitude = 0.1 

        for i in range(num_keyframes):
            #Adjust step size here 
            y_position = y_start - i * 0.01 
            x_shake = amplitude * np.sin(frequency * y_position) 
            z_position = 0.0 #can change to shake using z to get ingredients out of bowl

            position = X_WG_init.translation() + np.array([x_shake, y_position, z_position])
            rotation_matrix = X_WG_init.rotation()
            this_pose = RigidTransform(rotation_matrix, position)
            key_frame_poses_in_world.append(this_pose)

        times = np.linspace(start_time_s, start_time_s + move_time, len(key_frame_poses_in_world))
        trajectory = PiecewisePose.MakeLinear(times, key_frame_poses_in_world)
        return PiecewisePoseWithTimingInformation(
            trajectory=trajectory,
            start_time_s=start_time_s
        )
        



    #example use of function: 

    #gripper_action_example use
    # if self.gripper_action(context,current_time, state, "close") == True:
    #     transition_to_state(PizzaRobotState.FINISHED)

    #fix_base example:
    #new_joint_lims = self.calc_limits_fix_base_position(context, xyz_fixed=[0, 0, 1])
    #state.get_mutable_discrete_state().set_value(self._joint_velocity_limits_idx, new_joint_lims)

    #shimmy function example: 
    # y_start = 0  # Starting at y = 1 meter
    # num_keyframes = 20  # Generate 50 keyframes
    # amplitude = 0.1  # 20 cm total shake
    # frequency = 100  # Oscillate at 2 Hz
    # key_frame_poses = [X_WGinit] + shimmy_key_frames(y_start, num_keyframes, X_WCenter, amplitude, frequency)
    # times = np.linspace(0, total_time, num_keyframes + 1)
    # traj = PiecewisePose.MakeLinear(times, key_frame_poses)


