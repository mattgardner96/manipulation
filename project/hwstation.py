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



class PoseTrajectorySource(LeafSystem):
    """
    returns desired list of poses of dimension 20: 10 positions, 10 velocities
    (optional) pose_trajectory: trajectory to follow. if context does not already exist, pass it in from the plant.
    """
    # pose_trajectory: PiecewisePose = PiecewisePose()

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

def print_diagram(diagram, output_file="diagram.png"):
        # Visualize and save the diagram as a PNG
        graph = pydot.graph_from_dot_data(
            diagram.GetGraphvizString(max_depth=10)
        )[0]
        graph.write_png(output_file)
        print(f"Diagram saved to {output_file}")


def init_builder(meshcat, scenario, traj=PiecewisePose()):
    
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

    pos_to_state_sys = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(
            num_positions,
            time_step,
            suppress_initial_transient=True,
        )
    )

    traj_source = builder.AddSystem(PoseTrajectorySource(traj))

    builder.Connect(
        traj_source.get_output_port(),
        controller.get_input_port(0),
    )

    builder.Connect(
        station.GetOutputPort("mobile_iiwa.state_estimated"),
        controller.GetInputPort("robot_state"),
    )

    builder.Connect(
        controller.get_output_port(),
        pos_to_state_sys.get_input_port(),
    )

    builder.Connect(
        pos_to_state_sys.get_output_port(),
        station.GetInputPort("mobile_iiwa.desired_state")
    )

    return builder, station


def init_diagram(meshcat, scenario, traj=PiecewisePose()):
    builder, station = init_builder(meshcat, scenario, traj)

    diagram = builder.Build() # IMPT: must build the diagram before visualizing it
    diagram.set_name("diagram")

    simulator = Simulator(diagram)

    return diagram, simulator


# run simulation
def run_simulation(meshcat, simulator, start_time):
    # context = self.simulator.get_mutable_context()
    # x0 = self.station.GetOutputPort("mobile_iiwa.state_estimated").Eval(context)
    # self.station.GetInputPort("mobile_iiwa.desired_state").FixValue(context, x0)
    meshcat.StartRecording()
    simulator.AdvanceTo(start_time if running_as_notebook else 0.1)
    meshcat.PublishRecording()



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

    diagram,sim = init_diagram(meshcat,scenario)

    print_diagram(diagram)