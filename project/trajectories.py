def traj_linear_move_to_bowl_0(
        diagram,
        context=None,
        move_time=10 # seconds
    ) -> PoseTrajectorySource:
    if context is None:
        context = diagram.CreateDefaultContext()

    start_time = context.get_time()

    # get a starting point
    X_WGinit = hw.get_X_WG(diagram,context)

    # intermediate #1
    int_1 = hw.RigidTransform(
        R=hw.RotationMatrix(),
        p=np.array([0,1,1])
    )

    # intermediate #2
    int_2 = hw.RigidTransform(
        R=hw.RotationMatrix(),
        p=np.array([-4,1,1])
    )

    # define a goal pose
    goal_pose = hw.RigidTransform(
        R=hw.RotationMatrix(),
        p=np.array(env.bowl_0) + np.array([0.25, 0, 0.25])
    )

    poses = hw.PiecewisePose.MakeLinear([start_time,
        start_time + move_time/4,
        start_time + 3*move_time/4,
        start_time + move_time],
        [X_WGinit, int_1, int_2, goal_pose])

    return poses