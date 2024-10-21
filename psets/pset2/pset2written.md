## PSet 2 Written

Matt Gardner
6.4212
9/18/2024
Worked with Marine Maisonneuve, April Hu, and Arthur Pommersheim

### Exercise 2.1 b)

Similar to increasing Kp in a standard closed-loop system, the gearbox improves system performance by diminishing the impact of the unstable pendulum plant's dynamics on the output response. Classical control tells us that the in->out transfer function $\frac{Y(s)}{R(s)}$ of the closed-loop system simplifies to $\frac{GH}{1+GH}$; increasing N in the plant causes the GH term to dominate the expected response, so GH/(1+GH) approaches 1, dampening the effects of all other dynamics of the plant H to produce a linear response. In the direct-drive case, the system is much more sensitive to input condition (i.e. desired position) as the sinusoidal state-dependent term dominates the response and produces oscillation in the output. The gearbox case behaves much more like a $\frac{Y(s)}{R(s)} = 1$ condition, where the output dynamics closely mirror the input no matter what input we provide.

### Exercise 3.5 c)

The values for which there's no solution (kinematic singularity) are when $det(J)=0$. This occurs when the arm is fully extended ($q_1=0$) or when the arm is folded back on itself ($q_1=\pi$). These solutions are not dependent on $q_0$, so the first link can be at any angle.

Mathematically, $det(J)=ad-bc=0$ defines a kinematic singularity. In the case where $q_1=0$, $ad-bc = c(q_0)*(-2s(q_0)) + s(q_0)*2c(q_0) = 0.$  The algebra of the $q_1=\pi$ case is less trivial, but the same property applies; terms cancel such that the determinant is equal to 0. In these positions, a DOF is lost, so the proper joint angles cannot be unambiguously calculated for a given $^Ap^C$.

### Exercise 3.6

**a)** in the 2x2 Jacobian, we have a row for each DOF (XY) and a column for each joint angle (q0, q1). In the case described, we would have a 3xN matrix for the 3 DOFs (X,Y,Z) and N joint angles.

**b)** **i)** the inverse of the Jacobian can be computed when the determinant is not equal to 0, which implies that the matrix is invertible. In the case of the multi-link manipulator, this means that every joint has an angle besides 0˚ or 180˚ (nonsingular matrix, similar to preventing "gimbal lock" in a gyroscopic scenario). If the Jacobian can be inverted, there is precisely one set of joint angles which can produce the desired end effector velocity.

**ii)** if the dimensionality of the configuration space is larger than the dimensionality of the operational space, there are more than one set of joint velocities that can produce the desired end effector velocity. In this way, there are essentially more degrees of freedom of the arm than the arm requires to make the desired move in operational space. This occurs when the arm has more DOFs than the required DOFs of the end effector. Infinitely many sets can be commanded since the system is underdetermined.

**iii)** yes, in cases where the configuration space of the robot has fewer dimensions than those existing in the operational space. For example, picture a car with steering angle locked at 0˚ positioned along the x axis. The car can only move along the x-axis. However, say we want the car to move along the y-axis. Since it only has one degree of freedom, it cannot move in more than one dimension. Unlocking steering angle provides a second degree of freedom enabling travel in the y-axis. In a manipulator, this can also occur when the Jacobian loses rank due to a singularity (such as a fully extended arm) or the operational requirement is outside of the arm's reach.

**c)** at a singularity, the two-link manipulator has only 1 DOF. That means that the velocity of, say, $q_1$, is 0. Therefore, the matrix loses rank and the velocity of a single end-effector axis goes to zero as well. Thus, the manipulability ellipsoid collapses to a line segment where the arm can achieve a certain range of velocities in 1-dimensional space (along a line).

### Past Project Summary

I watched the Dreidel Spinning Robot video. Lily applies several different modeling/simulation techniques to first ensure that the dreidel appropriately spins as expected. Using the Kuka iiwa arm, Lily relaxes the velocity and position limits of the end effector joint ("wrist") to allow the robot to spin up the dreidel effectively. After the robot picks up the dreidel, the robot releases it and it spins accordingly. Finally, Lily investigates the Point Contact and Hydroelastic models of contact friction during dreidel spinup and compares the results.

### Survey

survey code is `iiwa kinematics`
