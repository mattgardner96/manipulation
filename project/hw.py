import os
import importlib
import sys

import numpy as np
from pydrake.all import (
    AbstractValue,
    AddDefaultVisualization,
    AddMultibodyPlantSceneGraph,
    ConstantVectorSource,
    DiagramBuilder,
    LeafSystem,
    LoadModelDirectives,
    LoadModelDirectivesFromString,
    Parser,
    PiecewisePose,
    ProcessModelDirectives,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    Simulator,
    StartMeshcat,
    LeafSystem,
    ConstantVectorSource,
    MultibodyPlant,
    Frame,
    DifferentialInverseKinematicsIntegrator,
    StateInterpolatorWithDiscreteDerivative,
    DifferentialInverseKinematicsParameters,
    LinearConstraint,
)

from IPython.display import display, SVG
import pydot
import matplotlib.pyplot as plt

from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from pydrake.visualization import ModelVisualizer

from manipulation import running_as_notebook
from manipulation.station import LoadScenario, MakeHardwareStation, MakeMultibodyPlant
from manipulation.utils import ConfigureParser
from manipulation.meshcat_utils import AddMeshcatTriad

sys.path.append('.')
import env_ingredient_add
importlib.reload(env_ingredient_add)
import trajectories
from trajectories import PizzaPlanner, PizzaRobotState


def CreateStateMachine(
    builder: DiagramBuilder, station, frame: Frame = None
):
    planner = PizzaPlanner(
        num_joint_positions=10,
        initial_delay_s=0,
        controller_plant=station.GetSubsystemByName("plant"),
    )

    # plant = station.GetSubsystemByName("plant")
    # plant_context = plant.CreateDefaultContext()
    # print(f"within CreateFSM: {plant.GetPositions(plant_context)}")
    state_machine = builder.AddNamedSystem("planner",planner)
    
    return state_machine


def CreateIiwaControllerPlant():
    #creates plant that includes only the robot and gripper, used for controllers
    robot_sdf_path = ("package://manipulation/mobile_iiwa14_primitive_collision.urdf")
    sim_timestep = 1e-3
    plant_robot = MultibodyPlant(sim_timestep)
    parser = Parser(plant=plant_robot)
    ConfigureParser(parser)
    parser.AddModelsFromUrl(robot_sdf_path)
    
    # Add gripper.
    parser.AddModelsFromUrl("package://manipulation/schunk_wsg_50_welded_fingers.dmd.yaml")
    plant_robot.WeldFrames(
        frame_on_parent_F=plant_robot.GetFrameByName("iiwa_link_7"),
        frame_on_child_M=plant_robot.GetFrameByName("body"),
        X_FM=RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2), [0, 0, 0.09])
    )
    
    plant_robot.mutable_gravity_field().set_gravity_vector([0, 0, 0])
    plant_robot.Finalize()
    return plant_robot

def AddMobileIiwaDifferentialIK(
    builder: DiagramBuilder, plant: MultibodyPlant, frame: Frame = None
) -> DifferentialInverseKinematicsIntegrator:
    """
    Args:
        builder: The DiagramBuilder to which the system should be added.

        plant: The MultibodyPlant passed to the DifferentialInverseKinematicsIntegrator.

        frame: The frame to use for the end effector command. Defaults to the body
            frame of "iiwa_link_7". NOTE: This must be present in the controller plant!

    Returns:
        The DifferentialInverseKinematicsIntegrator system.
    """
    assert plant.num_positions() == 10
    
    params = DifferentialInverseKinematicsParameters(
        plant.num_positions(), plant.num_velocities()   
    )
    time_step = plant.time_step()
    q0 = plant.GetPositions(plant.CreateDefaultContext())
    print(q0)
    params.set_nominal_joint_position(q0)
    params.set_end_effector_angular_speed_limit(2)
    params.set_end_effector_translational_velocity_limits([-2, -2, -2], [2, 2, 2])

    if frame is None:
        frame = plant.GetFrameByName("iiwa_link_7")

    mobile_iiwa_velocity_limits = np.array([0.5, 0.5, 0.5, 1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
    params.set_joint_velocity_limits(
        (-mobile_iiwa_velocity_limits, mobile_iiwa_velocity_limits)
    )
    params.set_joint_centering_gain(10 * np.eye(10))
    
    differential_ik = builder.AddNamedSystem(
        "diff_ik_integrator",
        DifferentialInverseKinematicsIntegrator(
            plant,
            frame,
            time_step,
            params,
            log_only_when_result_state_changes=True,
        )
    )
    return differential_ik


def diff_ik_solver(J_G, V_G_desired, q_now, v_now, p_now):
    num_positions = 10
    prog = MathematicalProgram()
    v = prog.NewContinuousVariables(10, "joint_velocities")
    v_max = 3.0  # do not modify
    h = 4e-3  # do not modify
    lower_bound = np.array([-0.3, -1.0, 0.0])  # do not modify
    upper_bound = np.array([0.3, 1.0, 1.0])  # do not modify

    p_next = (J_G @ v * h)[3:] + p_now

    # Fill in your code here.
    prog.AddCost((J_G @ v - V_G_desired).dot(J_G @ v - V_G_desired))
    prog.AddBoundingBoxConstraint(-v_max, v_max, v)
    prog.AddConstraint(le(lower_bound, p_next))
    prog.AddConstraint(ge(upper_bound, p_next))
    # prog.AddBoundingBoxConstraint(lower_bound, upper_bound, p_next)
    
    solver = SnoptSolver()
    result = solver.Solve(prog)

    if not (result.is_success()):
        raise ValueError("Could not find the optimal solution.")

    v_solution = result.GetSolution(v)
    return v_solution

def print_diagram(diagram, output_file="diagram.png"):
        # Visualize and save the diagram as a PNG
        graph = pydot.graph_from_dot_data(
            diagram.GetGraphvizString(max_depth=10)
        )[0]
        graph.write_png(output_file)
        print(f"Diagram saved to {output_file}")


def init_builder(meshcat, scenario, traj=PiecewisePose()):
    from trajectories import PoseTrajectorySource
    
    num_positions = 10
    time_step = 1e-3
    
    builder = DiagramBuilder()
    
    station = builder.AddSystem(MakeHardwareStation(
        scenario, meshcat, package_xmls=[os.getcwd() + "/package.xml"])
    )

    plant = station.GetSubsystemByName("plant")
    gripper_frame = plant.GetFrameByName("body")
    world_frame = plant.world_frame()

    iiwa_controller_plant = CreateIiwaControllerPlant()

    controller = AddMobileIiwaDifferentialIK(
        builder,
        plant=iiwa_controller_plant,
        frame=gripper_frame,
    )

    '''
    The integrator of the controller is seeded with a starting joint position array.
    I want to either: reset the diff IK with the current position of the robot (so it holds current position),
    or change the seed value to the current position of the robot.

    The more robust solution (that I'll need to do anyway) is to reset the diff IK and pause the integrator.

    Nicholas does this with a port controlling "reset_diff_ik" on the integrator. It's just a boolean.
    On the integrator side, he's got the integrator wrapped in a 
    '''

    pos_to_state_sys = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(
            num_positions,
            time_step,
            suppress_initial_transient=True,
        )
    )

    if traj is not None:
        traj_source = builder.AddSystem(PoseTrajectorySource(traj))

        # builder.Connect(
        #     traj_source.get_output_port(),
        #     controller.get_input_port(0),
        # )

    # builder.Connect(
    #     station.GetOutputPort("mobile_iiwa.state_estimated"),
    #     controller.GetInputPort("robot_state"),
    # )

    builder.Connect(
        controller.get_output_port(),
        pos_to_state_sys.get_input_port(),
    )

    builder.Connect(
        pos_to_state_sys.get_output_port(),
        station.GetInputPort("mobile_iiwa.desired_state")
    )

    state_machine = CreateStateMachine(builder,station)
    
    builder.Connect(
        state_machine.get_output_port(0),
        controller.get_input_port(0)
    )
    builder.Connect(
        station.GetOutputPort("mobile_iiwa.state_estimated"),
        state_machine.get_input_port(0)
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        state_machine.GetInputPort("body_poses")
    )

    return builder, station


def init_diagram(meshcat, scenario, traj=PiecewisePose()):
    builder, station = init_builder(meshcat, scenario, traj)

    diagram = builder.Build() # IMPT: must build the diagram before visualizing it
    diagram.set_name("diagram")

    simulator = Simulator(diagram)

    return diagram, simulator


# run simulation
def run_simulation(meshcat, simulator, start_time=0.1):
    """
    starts a simulation.
    meshcat: sim environment
    simulator: simulator provided from the needed diagram
    start_time: time (s)
    """
    # context = self.simulator.get_mutable_context()
    # x0 = self.station.GetOutputPort("mobile_iiwa.state_estimated").Eval(context)
    # self.station.GetInputPort("mobile_iiwa.desired_state").FixValue(context, x0)
    meshcat.StartRecording()
    simulator.AdvanceTo(start_time if running_as_notebook else 0.1)
    meshcat.PublishRecording()


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


def get_X_WG(diagram, context=None):
    if not context:
        context = diagram.CreateDefaultContext()
    plant = diagram.GetSubsystemByName("station").GetSubsystemByName("plant")
    plant_context = plant.GetMyMutableContextFromRoot(context)

    gripper_frame = plant.GetFrameByName("body")
    world_frame = plant.world_frame()

    X_WG = plant.CalcRelativeTransform(
        plant_context, frame_A=world_frame, frame_B=gripper_frame
    )
    return X_WG


def create_painter_trajectory(diagram,meshcat=None,context=None):
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

    X_WGinit = get_X_WG(diagram,context=context)
    # print("X_WGinit: ", X_WGinit)

    total_time = 20.0
    start_time = total_time 
    key_frame_poses = [X_WGinit] + compose_circular_key_frames(thetas, X_WCenter, radius)

    if meshcat is not None:
        for i, frame in enumerate(key_frame_poses):
            AddMeshcatTriad(meshcat, X_PT=frame, path="frame"+str(i))

    times = np.linspace(0, total_time, num_key_frames + 1)

    return PiecewisePose.MakeLinear(times, key_frame_poses)


def reset_positions(diff_ik_params: DifferentialInverseKinematicsParameters,q_set):
    new_joint_pos = q_set[:10]
    diff_ik_params.set_nominal_joint_position(q_set[:10])



def fix_base_pos(diff_ik_params: DifferentialInverseKinematicsParameters,fix_base):
    # example usage: set_base_vels(bot_ik_params,np.zeros((2,3))) 
    tolerance = 1e-2
    new_joint_pos = np.zeros((2,10))
    for i,axis in enumerate(fix_base):
        if axis:
            # if fixed, we set the position to existing
            new_joint_pos[:,i] = diff_ik_params.get_nominal_joint_position()[i]
        else:
            # if free, we set to infinity
            new_joint_pos[:, i] = np.array([[-np.inf],[np.inf]]).T
            print(new_joint_pos[:,i])
    # new_joint_vels[:,0:7] = curr_joint_vels(diff_ik_params)[:,0:7]
    new_joint_pos[:,3:10] = np.array([[-np.inf],[np.inf]]) * np.ones((2,7))
    diff_ik_params.set_joint_position_limits(tuple([new_joint_pos[0]-tolerance, new_joint_pos[1]+tolerance]))


if __name__=="__main__":
    print("hello")
    
    def get_scene():
        if os.getcwd() == "/datasets/_deepnote_work/manipulation/project": 
            #scene = open("/work/manipulation/project/objects/environment_setup.yaml")
            scene = env_ingredient_add.get_environment_set_up(no_scene=True,include_driver=True)
            xmls = [os.getcwd() + "/package.xml", "/work/manipulation/project/package.xml"]
        else:
            #scene = open("objects/environment_setup.yaml") # local setup
            scene = env_ingredient_add.get_environment_set_up(no_scene=True,include_driver=True)
            xmls = [os.getcwd() + "/package.xml"]
    
        return scene

    scenario = LoadScenario(data=get_scene())
    meshcat = StartMeshcat()

    # at present, trajectory blank doesn't work
    diagram,sim = init_diagram(meshcat,scenario)
    context = diagram.CreateDefaultContext()
    traj = create_small_trajectory(diagram,context)

    meshcat.Delete()
    diagram,sim = init_diagram(meshcat,scenario,traj)


    create_small_trajectory(diagram)
    run_simulation(meshcat,sim,20)

    # print_diagram(diagram)
