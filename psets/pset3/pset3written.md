# 6.4212 Pset 3 Written

### Matt Gardner / 6.4212 / 09-25-2024

**3.8 (b)**

1. Since we are only producing joint velocities, integration is essential to understand the current position of the end effector and all of the arm's links.
2. Initial state of the integrator is the constant term during the integration The integrator only calculates _relative_ position from the sum of the velocities commanded, so the integrator must be set to the beginning of the initial position of the arm's joints before moves are commanded.
3. State is set using the `robot_state` input port of the DiffInverseIK integrator. According to the documentation, _If the robot_state port is connected, then the initial state of the integrator is set to match the positions from this port (the port accepts the state vector with positions and velocities for easy of use with MultibodyPlant, but only the positions are used)._ This means that when we tie the input port to the robot state, the integrator assumes the initial state from the existing robot state at t0 automatically.

**3.12 (a)**

See handwritten

**3.12 (b)**

See handwritten

**3.12 (c)**

See handwritten

**3.13 (a)**

Using p = -π/2 equates to a rotation -90˚ along the y-axis. 

**3.13 (b)**


**3.13 (c)**
