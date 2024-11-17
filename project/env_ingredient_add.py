import numpy as np

num_mushrooms = 30
num_tomatoes = 30
bowl_0 = [-3, 0.5, 0.79]
bowl_1 = [-3, 1, 0.79]
bowl_2 = [-3, 1.5, 0.79]
pan_position = [-1, -0.5, 0.75]


robot_only = """
directives:
- add_model:
    name: mobile_iiwa
    file: package://manipulation/mobile_iiwa14_primitive_collision.urdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [ 1.6]
        iiwa_joint_7: [0]
        iiwa_base_x: [-1]
        iiwa_base_y: [1]
        iiwa_base_z: [0]
- add_model:
    name: gripper
    file: package://manipulation/schunk_wsg_50_welded_fingers.sdf

- add_weld:
    parent: mobile_iiwa::iiwa_link_7
    child: gripper::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90]}
"""

initial_scene = """
- add_model:
    name: pan1
    file: package://pizzabot/objects/panV1.sdf
    default_free_body_pose:
        panV1:
            translation: [-1, -0.5, 0.73]

- add_model:
    name: oven1
    file: package://pizzabot/objects/ovenV1.sdf

- add_model:
    name: serving
    file: package://pizzabot/objects/serving_shelf.sdf

- add_model:
    name: bowl0
    file: package://pizzabot/objects/mixing_bowl.sdf
    default_free_body_pose:
        mixing_bowl_body_link:
            translation: [-3, 0.5, 0.73]
            rotation: !Rpy { deg: [90.0, 0.0, 0.0]}

- add_model:
    name: bowl1
    file: package://pizzabot/objects/mixing_bowl.sdf
    default_free_body_pose:
        mixing_bowl_body_link:
            translation: [-3, 1, 0.73]
            rotation: !Rpy { deg: [90.0, 0.0, 0.0]}
- add_model:
    name: bowl2
    file: package://pizzabot/objects/mixing_bowl.sdf
    default_free_body_pose:
        mixing_bowl_body_link:
            translation: [-3, 1.5, 0.73]
            rotation: !Rpy { deg: [90.0, 0.0, 0.0]}
        
- add_model:
    name: table0
    file: package://pizzabot/objects/table.sdf

- add_model:
    name: table1
    file: package://pizzabot/objects/table.sdf

- add_model:
    name: camera0
    file: package://manipulation/camera_box.sdf

- add_frame:
    name: camera_table_above
    X_PF:
        base_frame: table0::table
        rotation: !Rpy { deg: [180.0, 0.0, 0.0]}
        translation: [0., 0., 1.5]

- add_weld:
    parent: camera_table_above
    child: camera0::base
    
- add_weld:
    parent: world
    child: table0::table
    X_PC:
      translation: [-1, -0.5, -0.015]

- add_weld:
    parent: world
    child: table1::table
    X_PC:
      translation: [-3, 1, -0.015]
      rotation: !Rpy { deg: [0.0, 0.0, -90.0 ]}

- add_weld:
    parent: world
    child: oven1::ovenV1
    X_PC:
      translation: [1, 1.5, -0.015]
      rotation: !Rpy { deg: [0.0, 0.0, -90.0 ]}

- add_weld:
    parent: world
    child: serving::serving_shelf
    X_PC:
      translation: [0, 2.5, -0.015]
      rotation: !Rpy { deg: [90.0, 0.0, 0.0 ]}


"""

model_drivers = """
model_drivers:
    mobile_iiwa: !InverseDynamicsDriver {}

"""  

def add_mushroom(scenario_data):

    mushroom_position = bowl_0 + np.array([-0.05,-0.05, 0.01])
    mushroom_instance = mushroom_position

    #to fix add rand number to sim to populate mushroom instances not on top of each other

    for i in range(num_mushrooms):
        if i%3 == 0:
            mushroom_instance = mushroom_instance + np.array([0, 0, 0.01]) 
        elif i%2 == 0: 
            mushroom_instance = mushroom_instance + np.array([0, 0.01, 0.0]) 
        else:
            mushroom_instance = mushroom_instance + np.array([0.01, 0.0, 0.0])
        scenario_data += """
- add_model:
    name: mushroom_"""+str(i)+"""
    file: package://pizzabot/objects/mush_slice.sdf
    default_free_body_pose:
        mush_slice:
            translation: ["""+str(mushroom_instance[0])+""", """+str(mushroom_instance[1])+""", """+str(mushroom_instance[2])+"""]
            
"""
    return scenario_data

def add_tomato(scenario_data):

    tomato_position = bowl_1 + np.array([-0.05,-0.05, 0])
    tomato_instance = tomato_position

    #to fix add rand number to sim to populate mushroom instances not on top of each other

    for j in range(num_tomatoes):
        if j%3 == 0:
            tomato_instance = tomato_instance + np.array([0, 0, 0.01]) 
        elif j%2 == 0: 
            tomato_instance = tomato_instance + np.array([0, 0.01, 0.0]) 
        else:
            tomato_instance = tomato_instance + np.array([0.01, 0.0, 0.0])
        scenario_data += """
- add_model:
    name: tomato_"""+str(j)+"""
    file: package://pizzabot/objects/tomato_slice.sdf
    default_free_body_pose:
        tomato_slice:
            translation: ["""+str(tomato_instance[0])+""", """+str(tomato_instance[1])+""", """+str(tomato_instance[2])+"""]
            
"""
    return scenario_data

def get_environment_set_up(no_scene=False,include_driver=True):

    scenario_data = robot_only
    if (no_scene == False):
        scenario_data += initial_scene
        scenario_data = add_mushroom(scenario_data)
        scenario_data = add_tomato(scenario_data)

    if include_driver:
        scenario_data += model_drivers
    
    return scenario_data
