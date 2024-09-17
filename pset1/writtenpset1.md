# Pset 1 Written
Matt Gardner, 6.4212, 9/10

### Question 1
1.1 Initial Condition for `x(0) = 0.9`

1.2 Angular Acceleration in ASCII: `alpha`

1.3 `TestControllers.options.convergence_tol = 1`

### Question 2
b. `torque_commanded = tau_no_ff + tau_ff.` As explained in the text, the iiwa arm's internal controller automatically maintains the commanded position and torque of each joint. The feedforward torque is mixed (added) to the torques required to maintain the motor's position in `position_and_torque` control mode. We specified position earlier on, and the controller is maintaining this position with no additional torque commands. When we command a feedforward torque, the result is the sum of the torque required for position hold + the commanded ff torque.

### Question 3
Video watched: `CatchingBot: A Robot that Catches Objects in Flight`
This project demonstrates an iiwa arm catching several objects thrown from 2-4 meters away. The robot is able to track the expected trajectory and pose of a target object, determine the optimal catch angle, and constantly resolve the optimal gripper pose in space. Using a constrained optimization problem, the team then plans the optimal robot path to reach the desired gripper location. During the implementation of the project, the team encountered many issues with trajectory tracking and gripper timing. Trajectory issues were resolved, but the system was still sensitive to gripper close timing at the conclusion of the project.

### Question 4
Code is `HardwareStation`