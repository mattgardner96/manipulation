class PizzaBot:
    # make robot station and set up simulation
    def __init__(self, scenario=None, traj=None):
        self.meshcat = meshcat
        builder = DiagramBuilder()
        
        self.station = builder.AddSystem(MakeHardwareStation(
            scenario, meshcat, package_xmls=[os.getcwd() + "/package.xml"])
        )
        
        # add plant (iiwa arm)
        self.plant = self.station.GetSubsystemByName("plant")
        self.plant.GetJointByName("iiwa_joint_7").set_position_limits(
            [-np.inf], [np.inf]
        )
        
        # self.diagram = builder.Build()
        self.gripper_frame = self.plant.GetFrameByName("body")
        self.world_frame = self.plant.world_frame()


        # optionally add trajectory source
        if traj is not None:
            traj_source = builder.AddSystem(PoseTrajectorySource(traj))
            builder.Connect(
                traj_source.get_output_port(),
                self.controller.get_input_port(0),
            )
            builder.Connect(
                self.station.GetOutputPort("iiwa.state_estimated"),
                self.controller.GetInputPort("robot_state"),
            )
            builder.Connect(
                self.controller.get_output_port(),
                self.station.GetInputPort("iiwa.position"),
            )
        else:
            iiwa_position = builder.AddSystem(ConstantVectorSource(np.zeros(7)))
            builder.Connect(
                iiwa_position.get_output_port(),
                self.station.GetInputPort("iiwa.position"),
            )

        wsg_position = builder.AddSystem(ConstantVectorSource([0.1]))
        builder.Connect(
            wsg_position.get_output_port(),
            self.station.GetInputPort("wsg.position"),
        )

        self.diagram = builder.Build() # IMPT: must build the diagram before visualizing it
        self.diagram.set_name("diagram")

        self.simulator = Simulator(self.station)
    
        # BEGIN OLD CODE
        
        # self.plant.SetDefaultPositions(np.zeros(self.plant.num_positions()))
        # plant_context = self.plant.GetMyContextFromRoot(self.simulator.get_mutable_context())

        # controller_plant = self.station.GetSubsystemByName(
        #     "iiwa_controller_plant_pointer_system",
        # ).get()

    def print_diagram(self):
        print("station.system names: ")
        for sys in self.station.GetSystems():
            print("- " + sys.GetSystemName())

        # visualize the diagram, taken from pydrake tutorials
        display(SVG(pydot.graph_from_dot_data(
        self.diagram.GetGraphvizString(max_depth=2))[0].create_svg()))

    


    def get_X_WG(self, context=None):
        if not context:
            context = self.CreateDefaultContext()
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        X_WG = self.plant.CalcRelativeTransform(
            plant_context, frame_A=self.world_frame, frame_B=self.gripper_frame
        )
        return X_WG

    def CreateDefaultContext(self):
        context = self.diagram.CreateDefaultContext()

        # provide initial states
        q0 = np.array(
            [
                1.40666193e-05,
                1.56461165e-01,
                -3.82761069e-05,
                -1.32296976e00,
                -6.29097287e-06,
                1.61181157e00,
                -2.66900985e-05,
            ]
        )
        # set the joint positions of the kuka arm
        iiwa = self.plant.GetModelInstanceByName("mobile_iiwa")
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        self.plant.SetPositions(plant_context, iiwa, q0)
        self.plant.SetVelocities(plant_context, iiwa, np.zeros(7))

        wsg = self.plant.GetModelInstanceByName("gripper")
        self.plant.SetPositions(plant_context, gripper, [-0.05, 0.05])
        self.plant.SetVelocities(plant_context, gripper, [0, 0])

        return context

    # run simulation
    def run_simulation(self, start_time):
        context = self.simulator.get_mutable_context()
        x0 = self.station.GetOutputPort("mobile_iiwa.state_estimated").Eval(context)
        self.station.GetInputPort("mobile_iiwa.desired_state").FixValue(context, x0)
        self.meshcat.StartRecording()
        self.simulator.AdvanceTo(start_time if running_as_notebook else 0.1)
        self.meshcat.PublishRecording()