import math
import os
import random
import sys

import git
import magnum as mn
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as ut
from habitat_sim.utils import viz_utils as vut

try:
    import ipywidgets as widgets
    from IPython.display import display as ipydisplay

    # For using jupyter/ipywidget IO components

    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False


if "google.colab" in sys.modules:
    os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir

data_path = os.path.join(dir_path, "data")
output_directory = "examples/tutorials/interactivity_output/"  # @param {type:"string"}
output_path = os.path.join(dir_path, output_directory)
if not os.path.exists(output_path):
    os.mkdir(output_path)

# define some globals the first time we run.
if "sim" not in globals():
    global sim
    sim = None
    global obj_attr_mgr
    obj_attr_mgr = None
    global prim_attr_mgr
    prim_attr_mgr = None
    global stage_attr_mgr
    stage_attr_mgr = None
    global rigid_obj_mgr
    rigid_obj_mgr = None


#===========================================================
# @title Define Configuration Utility Functions

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    # Optional; Specify the location of an existing scene dataset configuration
    # that describes the locations and configurations of all the assets to be used
    if "scene_dataset_config" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config"]

    # Note: all sensors must have the same resolution
    sensor_specs = []
    if settings["color_sensor_1st_person"]:
        color_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
        color_sensor_1st_person_spec.uuid = "color_sensor_1st_person"
        color_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_1st_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        color_sensor_1st_person_spec.position = [0.0, settings["sensor_height"], 0.0]
        color_sensor_1st_person_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        color_sensor_1st_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_1st_person_spec)
    if settings["depth_sensor_1st_person"]:
        depth_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_1st_person_spec.uuid = "depth_sensor_1st_person"
        depth_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_1st_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        depth_sensor_1st_person_spec.position = [0.0, settings["sensor_height"], 0.0]
        depth_sensor_1st_person_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        depth_sensor_1st_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(depth_sensor_1st_person_spec)
    if settings["semantic_sensor_1st_person"]:
        semantic_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_1st_person_spec.uuid = "semantic_sensor_1st_person"
        semantic_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_1st_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        semantic_sensor_1st_person_spec.position = [
            0.0,
            settings["sensor_height"],
            0.0,
        ]
        semantic_sensor_1st_person_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        semantic_sensor_1st_person_spec.sensor_subtype = (
            habitat_sim.SensorSubType.PINHOLE
        )
        sensor_specs.append(semantic_sensor_1st_person_spec)
    if settings["color_sensor_3rd_person"]:
        color_sensor_3rd_person_spec = habitat_sim.CameraSensorSpec()
        color_sensor_3rd_person_spec.uuid = "color_sensor_3rd_person"
        color_sensor_3rd_person_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_3rd_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        color_sensor_3rd_person_spec.position = [
            0.0,
            settings["sensor_height"] + 1.2,
            0.2,
        ]
        color_sensor_3rd_person_spec.orientation = [-math.pi / 4, 0, 0]
        color_sensor_3rd_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_3rd_person_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def make_default_settings():
    settings = {
        "width": 720,  # Spatial resolution of the observations
        "height": 544,
        "scene": "./data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb",  # Scene path
        "default_agent": 0,
        "sensor_height": 1.5,  # Height of sensors in meters
        "sensor_pitch": -math.pi / 8.0,  # sensor pitch (x rotation in rads)
        "color_sensor_1st_person": True,  # RGB sensor
        "color_sensor_3rd_person": True,  # RGB sensor 3rd person
        "depth_sensor_1st_person": True,  # Depth sensor
        "semantic_sensor_1st_person": False,  # Semantic sensor
        "seed": 1,
        "enable_physics": True,  # enable dynamics simulation
    }
    return settings


def make_simulator_from_settings(sim_settings):
    cfg = make_cfg(sim_settings)
    # clean-up the current simulator instance if it exists
    global sim
    global obj_attr_mgr
    global prim_attr_mgr
    global stage_attr_mgr
    global rigid_obj_mgr
    if sim != None:
        sim.close()
    # initialize the simulator
    sim = habitat_sim.Simulator(cfg)
    # Managers of various Attributes templates
    obj_attr_mgr = sim.get_object_template_manager()
    obj_attr_mgr.load_configs(str(os.path.join(data_path, "objects/example_objects")))
    obj_attr_mgr.load_configs(str(os.path.join(data_path, "objects/locobot_merged")))
    prim_attr_mgr = sim.get_asset_template_manager()
    stage_attr_mgr = sim.get_stage_template_manager()
    # Manager providing access to rigid objects
    rigid_obj_mgr = sim.get_rigid_object_manager()

#===========================================================
# Define Simulation Utility Functions { display-mode: "form" }

def simulate(sim, dt=1.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())
    return observations


# Set an object transform relative to the agent state
def set_object_state_from_agent(
    sim,
    obj,
    offset=np.array([0, 2.0, -1.5]),
    orientation=mn.Quaternion(((0, 0, 0), 1)),
):
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    ob_translation = agent_transform.transform_point(offset)
    obj.translation = ob_translation
    obj.rotation = orientation


# sample a random valid state for the object from the scene bounding box or navmesh
def sample_object_state(
    sim, obj, from_navmesh=True, maintain_object_up=True, max_tries=100, bb=None
):
    # check that the object is not STATIC
    if obj.motion_type is habitat_sim.physics.MotionType.STATIC:
        print("sample_object_state : Object is STATIC, aborting.")
    if from_navmesh:
        if not sim.pathfinder.is_loaded:
            print("sample_object_state : No pathfinder, aborting.")
            return False
    elif not bb:
        print(
            "sample_object_state : from_navmesh not specified and no bounding box provided, aborting."
        )
        return False
    tries = 0
    valid_placement = False
    # Note: following assumes sim was not reconfigured without close
    scene_collision_margin = stage_attr_mgr.get_template_by_id(0).margin
    while not valid_placement and tries < max_tries:
        tries += 1
        # initialize sample location to random point in scene bounding box
        sample_location = np.array([0, 0, 0])
        if from_navmesh:
            # query random navigable point
            sample_location = sim.pathfinder.get_random_navigable_point()
        else:
            sample_location = np.random.uniform(bb.min, bb.max)
        # set the test state
        obj.translation = sample_location
        if maintain_object_up:
            # random rotation only on the Y axis
            y_rotation = mn.Quaternion.rotation(
                mn.Rad(random.random() * 2 * math.pi), mn.Vector3(0, 1.0, 0)
            )
            obj.rotation = y_rotation * obj.rotation
        else:
            # unconstrained random rotation
            obj.rotation = ut.random_quaternion()

        # raise object such that lowest bounding box corner is above the navmesh sample point.
        if from_navmesh:
            obj_node = obj.root_scene_node
            xform_bb = habitat_sim.geo.get_transformed_bb(
                obj_node.cumulative_bb, obj_node.transformation
            )
            # also account for collision margin of the scene
            obj.translation += mn.Vector3(
                0, xform_bb.size_y() / 2.0 + scene_collision_margin, 0
            )

        # test for penetration with the environment
        if not sim.contact_test(obj.object_id):
            valid_placement = True

    if not valid_placement:
        return False
    return True

# ==============================================================
# Define Visualization Utility Function { display-mode: "form" }

def display_sample(
    rgb_1st_obs, rgb_3rd_obs=np.array([]), semantic_obs=np.array([]), depth_obs=np.array([]), key_points=None
):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_1st_img = Image.fromarray(rgb_1st_obs, mode="RGBA")

    arr = [rgb_1st_img]
    titles = ["rgb_1st"]

    if rgb_3rd_obs.size != 0:
        rgb_3rd_img = Image.fromarray(rgb_3rd_obs, mode="RGBA")
        arr.append(rgb_3rd_img)
        titles.append("rgb_3rd")

    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        # plot points on images
        if key_points is not None:
            for point in key_points:
                plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
        plt.imshow(data)

    plt.show(block=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--no-make-video", dest="make_video", action="store_false")
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()
    show_video = args.display
    display = args.display
    make_video = args.make_video
else:
    show_video = False
    make_video = False
    display = False


#====================================================================
# Define Colab GUI Utility Functions


# Event handler for dropdowns displaying file-based object handles
def on_file_obj_ddl_change(ddl_values):
    global sel_file_obj_handle
    sel_file_obj_handle = ddl_values["new"]
    return sel_file_obj_handle


# Event handler for dropdowns displaying prim-based object handles
def on_prim_obj_ddl_change(ddl_values):
    global sel_prim_obj_handle
    sel_prim_obj_handle = ddl_values["new"]
    return sel_prim_obj_handle


# Event handler for dropdowns displaying asset handles
def on_prim_ddl_change(ddl_values):
    global sel_asset_handle
    sel_asset_handle = ddl_values["new"]
    return sel_asset_handle


# Build a dropdown list holding obj_handles and set its event handler
def set_handle_ddl_widget(obj_handles, handle_types, sel_handle, on_change):
    sel_handle = obj_handles[0]
    descStr = handle_types + " Template Handles:"
    style = {"description_width": "300px"}
    obj_ddl = widgets.Dropdown(
        options=obj_handles,
        value=sel_handle,
        description=descStr,
        style=style,
        disabled=False,
        layout={"width": "max-content"},
    )

    obj_ddl.observe(on_change, names="value")
    return obj_ddl, sel_handle


def set_button_launcher(desc):
    button = widgets.Button(
        description=desc,
        layout={"width": "max-content"},
    )
    return button


def make_sim_and_vid_button(prefix, dt=1.0):
    if not HAS_WIDGETS:
        return

    def on_sim_click(b):
        observations = simulate(sim, dt=dt)
        vut.make_video(
            observations, "color_sensor_1st_person", "color", output_path + prefix
        )

    sim_and_vid_btn = set_button_launcher("Simulate and Make Video")
    sim_and_vid_btn.on_click(on_sim_click)
    ipydisplay(sim_and_vid_btn)


def make_clear_all_objects_button():
    if not HAS_WIDGETS:
        return

    def on_clear_click(b):
        rigid_obj_mgr.remove_all_objects()

    clear_objs_button = set_button_launcher("Clear all objects")
    clear_objs_button.on_click(on_clear_click)
    ipydisplay(clear_objs_button)


# Builds widget-based UI components
def build_widget_ui(obj_attr_mgr, prim_attr_mgr):
    # Holds the user's desired file-based object template handle
    global sel_file_obj_handle
    sel_file_obj_handle = ""

    # Holds the user's desired primitive-based object template handle
    global sel_prim_obj_handle
    sel_prim_obj_handle = ""

    # Holds the user's desired primitive asset template handle
    global sel_asset_handle
    sel_asset_handle = ""

    # Construct DDLs and assign event handlers
    # All file-based object template handles
    file_obj_handles = obj_attr_mgr.get_file_template_handles()
    prim_obj_handles = obj_attr_mgr.get_synth_template_handles()
    prim_asset_handles = prim_attr_mgr.get_template_handles()
    if not HAS_WIDGETS:
        sel_file_obj_handle = file_obj_handles[0]
        sel_prim_obj_handle = prim_obj_handles[0]
        sel_asset_handle = prim_asset_handles[0]
        return
    file_obj_ddl, sel_file_obj_handle = set_handle_ddl_widget(
        file_obj_handles,
        "File-based Object",
        sel_file_obj_handle,
        on_file_obj_ddl_change,
    )
    # All primitive asset-based object template handles
    prim_obj_ddl, sel_prim_obj_handle = set_handle_ddl_widget(
        prim_obj_handles,
        "Primitive-based Object",
        sel_prim_obj_handle,
        on_prim_obj_ddl_change,
    )
    # All primitive asset handles template handles
    prim_asset_ddl, sel_asset_handle = set_handle_ddl_widget(
        prim_asset_handles, "Primitive Asset", sel_asset_handle, on_prim_ddl_change
    )
    # Display DDLs
    ipydisplay(file_obj_ddl)
    ipydisplay(prim_obj_ddl)
    ipydisplay(prim_asset_ddl)


#===========================================================================
# Initialize Simulator and Load Scene { display-mode: "form" }

# convienience functions defined in Utility cell manage global variables
sim_settings = make_default_settings()
# set globals: sim,
make_simulator_from_settings(sim_settings)

#=============================================================================
# Select a Simulation Object Template
build_widget_ui(obj_attr_mgr, prim_attr_mgr)

# @title Add either a File-based or Primitive Asset-based object to the scene at a user-specified location.{ display-mode: "form" }
# @markdown Running this will add a physically-modelled object of the selected type to the scene at the location specified by user, simulate forward for a few seconds and save a movie of the results.

# @markdown Choose either the primitive or file-based template recently selected in the dropdown:
obj_template_handle = sel_file_obj_handle
asset_tempalte_handle = sel_asset_handle
object_type = "File-based"  # @param ["File-based","Primitive-based"]
if "File" in object_type:
    # Handle File-based object handle
    obj_template_handle = sel_file_obj_handle
elif "Primitive" in object_type:
    # Handle Primitive-based object handle
    obj_template_handle = sel_prim_obj_handle
else:
    # Unknown - defaults to file-based
    pass

# @markdown Configure the initial object location (local offset from the agent body node):
# default : offset=np.array([0,2.0,-1.5]), orientation=np.quaternion(1,0,0,0)
offset_x = 0.5  # @param {type:"slider", min:-2, max:2, step:0.1}
offset_y = 1.4  # @param {type:"slider", min:0, max:3.0, step:0.1}
offset_z = -1.5  # @param {type:"slider", min:-3, max:0, step:0.1}
offset = np.array([offset_x, offset_y, offset_z])

# @markdown Configure the initial object orientation via local Euler angle (degrees):
orientation_x = 0  # @param {type:"slider", min:-180, max:180, step:1}
orientation_y = 0  # @param {type:"slider", min:-180, max:180, step:1}
orientation_z = 0  # @param {type:"slider", min:-180, max:180, step:1}

# compose the rotations
rotation_x = mn.Quaternion.rotation(mn.Deg(orientation_x), mn.Vector3(1.0, 0, 0))
rotation_y = mn.Quaternion.rotation(mn.Deg(orientation_y), mn.Vector3(1.0, 0, 0))
rotation_z = mn.Quaternion.rotation(mn.Deg(orientation_z), mn.Vector3(1.0, 0, 0))
orientation = rotation_z * rotation_y * rotation_x

# Add object instantiated by desired template using template handle
obj_1 = rigid_obj_mgr.add_object_by_template_handle(obj_template_handle)

# @markdown Note: agent local coordinate system is Y up and -Z forward.
# Move object to be in front of the agent
set_object_state_from_agent(sim, obj_1, offset=offset, orientation=orientation)

# display a still frame of the scene after the object is added if RGB sensor is enabled
observations = sim.get_sensor_observations()
if display: 
    rgb_1st_obs = observations["color_sensor_1st_person"] if sim_settings["color_sensor_1st_person"] else np.array([])
    rgb_3rd_obs = observations["color_sensor_3rd_person"] if sim_settings["color_sensor_3rd_person"] else np.array([])
    depth_obs = observations["depth_sensor_1st_person"] if sim_settings["depth_sensor_1st_person"] else np.array([])
    
    display_sample(rgb_1st_obs, rgb_3rd_obs, np.array([]), depth_obs)

example_type = "adding objects test"
make_sim_and_vid_button(example_type)
make_clear_all_objects_button()


# @title Select object templates from the GUI: { display-mode: "form" }
build_widget_ui(obj_attr_mgr, prim_attr_mgr)

# @title Scripted vs. Dynamic Motion { display-mode: "form" }
# @markdown A quick script to generate video data for AI classification of dynamically dropping vs. kinematically moving objects.
rigid_obj_mgr.remove_all_objects()
# @markdown Set the scene as dynamic or kinematic:
scenario_is_kinematic = True  # @param {type:"boolean"}

# add the selected object
obj_1 = rigid_obj_mgr.add_object_by_template_handle(sel_file_obj_handle)

# place the object
set_object_state_from_agent(
    sim, obj_1, offset=np.array([0, 2.0, -1.0]), orientation=ut.random_quaternion()
)

if scenario_is_kinematic:
    # use the velocity control struct to setup a constant rate kinematic motion
    obj_1.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    vel_control = obj_1.velocity_control
    vel_control.controlling_lin_vel = True
    vel_control.linear_velocity = np.array([0, -1.0, 0.0])

# simulate and collect observations
example_type = "kinematic vs dynamic"
observations = simulate(sim, dt=2.0)
if make_video:
    vut.make_video(
        observations,
        "color_sensor_1st_person",
        "color",
        output_path + example_type,
        open_vid=show_video,
    )

rigid_obj_mgr.remove_all_objects()

# @title Object Permanence { display-mode: "form" }
# @markdown This example script demonstrates a possible object permanence task.
# @markdown Two objects are dropped behind an occluder. One is removed while occluded.
rigid_obj_mgr.remove_all_objects()

# @markdown 1. Add the two dynamic objects.
# add the selected objects
obj_1 = rigid_obj_mgr.add_object_by_template_handle(sel_file_obj_handle)
obj_2 = rigid_obj_mgr.add_object_by_template_handle(sel_file_obj_handle)

# place the objects
set_object_state_from_agent(
    sim, obj_1, offset=np.array([0.5, 2.0, -1.0]), orientation=ut.random_quaternion()
)
set_object_state_from_agent(
    sim,
    obj_2,
    offset=np.array([-0.5, 2.0, -1.0]),
    orientation=ut.random_quaternion(),
)

# @markdown 2. Configure and add an occluder from a scaled cube primitive.
# Get a default cube primitive template
cube_handle = obj_attr_mgr.get_template_handles("cube")[0]
cube_template_cpy = obj_attr_mgr.get_template_by_handle(cube_handle)
# Modify the template's configured scale.
cube_template_cpy.scale = np.array([0.32, 0.075, 0.01])
# Register the modified template under a new name.
obj_attr_mgr.register_template(cube_template_cpy, "occluder_cube")
# Instance and place the occluder object from the template.
occluder_obj = rigid_obj_mgr.add_object_by_template_handle("occluder_cube")
set_object_state_from_agent(sim, occluder_obj, offset=np.array([0.0, 1.4, -0.4]))
occluder_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
# fmt off
# @markdown 3. Simulate at 60Hz, removing one object when it's center of mass drops below that of the occluder.
# fmt on
# Simulate and remove object when it passes the midpoint of the occluder
dt = 2.0
print("Simulating " + str(dt) + " world seconds.")
observations = []
# simulate at 60Hz to the nearest fixed timestep
start_time = sim.get_world_time()

while sim.get_world_time() < start_time + dt:
    sim.step_physics(1.0 / 60.0)
    # remove the object once it passes the occluder center and it still exists/hasn't already been removed
    if obj_2.is_alive and obj_2.translation[1] <= occluder_obj.translation[1]:
        rigid_obj_mgr.remove_object_by_id(obj_2.object_id)
    observations.append(sim.get_sensor_observations())

example_type = "object permanence"
if make_video:
    vut.make_video(
        observations,
        "color_sensor_1st_person",
        "color",
        output_path + example_type,
        open_vid=show_video,
    )
rigid_obj_mgr.remove_all_objects()


# @title Physical Plausibility Classification { display-mode: "form" }
# @markdown This example demonstrates a physical plausibility expirement. A sphere
# @markdown is dropped onto the back of a couch to roll onto the floor. Optionally,
# @markdown an invisible plane is introduced for the sphere to roll onto producing
# @markdown non-physical motion.

introduce_surface = True  # @param{type:"boolean"}

rigid_obj_mgr.remove_all_objects()

# add a rolling object
obj_attr_mgr = sim.get_object_template_manager()
sphere_handle = obj_attr_mgr.get_template_handles("uvSphereSolid")[0]
obj_1 = rigid_obj_mgr.add_object_by_template_handle(sphere_handle)
set_object_state_from_agent(sim, obj_1, offset=np.array([1.0, 1.6, -1.95]))

if introduce_surface:
    # optionally add invisible surface
    cube_handle = obj_attr_mgr.get_template_handles("cube")[0]
    cube_template_cpy = obj_attr_mgr.get_template_by_handle(cube_handle)
    # Modify the template.
    cube_template_cpy.scale = np.array([1.0, 0.04, 1.0])
    surface_is_visible = False  # @param{type:"boolean"}
    cube_template_cpy.is_visibile = surface_is_visible
    # Register the modified template under a new name.
    obj_attr_mgr.register_template(cube_template_cpy, "invisible_surface")

    # Instance and place the surface object from the template.
    surface_obj = rigid_obj_mgr.add_object_by_template_handle("invisible_surface")
    set_object_state_from_agent(sim, surface_obj, offset=np.array([0.4, 0.88, -1.6]))
    surface_obj.motion_type = habitat_sim.physics.MotionType.STATIC


example_type = "physical plausibility"
observations = simulate(sim, dt=3.0)
if make_video:
    vut.make_video(
        observations,
        "color_sensor_1st_person",
        "color",
        output_path + example_type,
        open_vid=show_video,
    )
rigid_obj_mgr.remove_all_objects()



# @title Trajectory Prediction { display-mode: "form" }
# @markdown This example demonstrates setup of a trajectory prediction task.
# @markdown Boxes are placed in a target zone and a sphere is given an initial
# @markdown velocity with the goal of knocking the boxes off the counter.

# @markdown ---
# @markdown Configure Parameters:

rigid_obj_mgr.remove_all_objects()

seed = 2  # @param{type:"integer"}
random.seed(seed)
sim.seed(seed)
np.random.seed(seed)

# setup agent state manually to face the bar
agent_state = sim.agents[0].state
agent_state.position = np.array([-1.97496, 0.072447, -2.0894])
agent_state.rotation = ut.quat_from_coeffs([0, -1, 0, 0])
sim.agents[0].set_state(agent_state)

# load the target objects
cheezit_handle = obj_attr_mgr.get_template_handles("cheezit")[0]
# create range from center and half-extent
target_zone = mn.Range3D.from_center(
    mn.Vector3(-2.07496, 1.07245, -0.2894), mn.Vector3(0.5, 0.05, 0.1)
)
num_targets = 9  # @param{type:"integer"}
for _target in range(num_targets):
    obj = rigid_obj_mgr.add_object_by_template_handle(cheezit_handle)
    # rotate boxes off of their sides
    obj.rotation = mn.Quaternion.rotation(
        mn.Rad(-mn.math.pi_half), mn.Vector3(1.0, 0, 0)
    )
    # sample state from the target zone
    if not sample_object_state(sim, obj, False, True, 100, target_zone):
        rigid_obj_mgr.remove_object_by_id(obj.object_id)


show_target_zone = False  # @param{type:"boolean"}
if show_target_zone:
    # Get and modify the wire cube template from the range
    cube_handle = obj_attr_mgr.get_template_handles("cubeWireframe")[0]
    cube_template_cpy = obj_attr_mgr.get_template_by_handle(cube_handle)
    cube_template_cpy.scale = target_zone.size()
    cube_template_cpy.is_collidable = False
    # Register the modified template under a new name.
    obj_attr_mgr.register_template(cube_template_cpy, "target_zone")
    # instance and place the object from the template
    target_zone_obj = rigid_obj_mgr.add_object_by_template_handle("target_zone")
    target_zone_obj.translation = target_zone.center()
    target_zone_obj.motion_type = habitat_sim.physics.MotionType.STATIC
    # print("target_zone_center = " + str(target_zone_obj.translation))

# @markdown ---
# @markdown ###Ball properties:
# load the ball
sphere_handle = obj_attr_mgr.get_template_handles("uvSphereSolid")[0]
sphere_template_cpy = obj_attr_mgr.get_template_by_handle(sphere_handle)
# @markdown Mass:
ball_mass = 5.01  # @param {type:"slider", min:0.01, max:50.0, step:0.01}
sphere_template_cpy.mass = ball_mass
obj_attr_mgr.register_template(sphere_template_cpy, "ball")

ball_obj = rigid_obj_mgr.add_object_by_template_handle("ball")
set_object_state_from_agent(sim, ball_obj, offset=np.array([0, 1.4, 0]))

# @markdown Initial linear velocity (m/sec):
lin_vel_x = 0  # @param {type:"slider", min:-10, max:10, step:0.1}
lin_vel_y = 1  # @param {type:"slider", min:-10, max:10, step:0.1}
lin_vel_z = 5  # @param {type:"slider", min:0, max:10, step:0.1}
ball_obj.linear_velocity = mn.Vector3(lin_vel_x, lin_vel_y, lin_vel_z)

# @markdown Initial angular velocity (rad/sec):
ang_vel_x = 0  # @param {type:"slider", min:-100, max:100, step:0.1}
ang_vel_y = 0  # @param {type:"slider", min:-100, max:100, step:0.1}
ang_vel_z = 0  # @param {type:"slider", min:-100, max:100, step:0.1}
ball_obj.angular_velocity = mn.Vector3(ang_vel_x, ang_vel_y, ang_vel_z)

example_type = "trajectory prediction"
observations = simulate(sim, dt=3.0)
if make_video:
    vut.make_video(
        observations,
        "color_sensor_1st_person",
        "color",
        output_path + example_type,
        open_vid=show_video,
    )
rigid_obj_mgr.remove_all_objects()


# ===============================================================
#  Generating Scene Clutter on the NavMesh

# @title Initialize Simulator and Load Scene { display-mode: "form" }
# @markdown (load the apartment_1 scene for clutter generation in an open space)
sim_settings = make_default_settings()
sim_settings["scene"] = "./data/scene_datasets/habitat-test-scenes/apartment_1.glb"
sim_settings["sensor_pitch"] = 0

make_simulator_from_settings(sim_settings)

build_widget_ui(obj_attr_mgr, prim_attr_mgr)

# @title Clutter Generation Script
# @markdown Configure some example parameters:

seed = 2  # @param {type:"integer"}
random.seed(seed)
sim.seed(seed)
np.random.seed(seed)

# position the agent
sim.agents[0].scene_node.translation = mn.Vector3(0.5, -1.60025, 6.15)
print(sim.agents[0].scene_node.rotation)
agent_orientation_y = -23  # @param{type:"integer"}
sim.agents[0].scene_node.rotation = mn.Quaternion.rotation(
    mn.Deg(agent_orientation_y), mn.Vector3(0, 1.0, 0)
)

num_objects = 10  # @param {type:"slider", min:0, max:20, step:1}
object_scale = 5  # @param {type:"slider", min:1.0, max:10.0, step:0.1}

# scale up the selected object
sel_obj_template_cpy = obj_attr_mgr.get_template_by_handle(sel_file_obj_handle)
sel_obj_template_cpy.scale = mn.Vector3(object_scale)
obj_attr_mgr.register_template(sel_obj_template_cpy, "scaled_sel_obj")

# add the selected object
sim.navmesh_visualization = True
rigid_obj_mgr.remove_all_objects()
fails = 0
for _obj in range(num_objects):
    obj_1 = rigid_obj_mgr.add_object_by_template_handle("scaled_sel_obj")

    # place the object
    placement_success = sample_object_state(
        sim, obj_1, from_navmesh=True, maintain_object_up=True, max_tries=100
    )
    if not placement_success:
        fails += 1
        rigid_obj_mgr.remove_object_by_id(obj_1.object_id)
    else:
        # set the objects to STATIC so they can be added to the NavMesh
        obj_1.motion_type = habitat_sim.physics.MotionType.STATIC

print("Placement fails = " + str(fails) + "/" + str(num_objects))

# recompute the NavMesh with STATIC objects
navmesh_settings = habitat_sim.NavMeshSettings()
navmesh_settings.set_defaults()
navmesh_success = sim.recompute_navmesh(
    sim.pathfinder, navmesh_settings, include_static_objects=True
)

# simulate and collect observations
example_type = "clutter generation"
observations = simulate(sim, dt=2.0)
if make_video:
    vut.make_video(
        observations,
        "color_sensor_1st_person",
        "color",
        output_path + example_type,
        open_vid=show_video,
    )
obj_attr_mgr.remove_template_by_handle("scaled_sel_obj")
rigid_obj_mgr.remove_all_objects()
sim.navmesh_visualization = False

