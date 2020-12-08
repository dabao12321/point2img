# TODO: Change this to the three camera setup.  Make sure to pull images from both bins.

# TODO: Compute mean and std for dataset and add the normalization transform.
# https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198

# TODO: Check if I'm running as a unit test and only do a single image.

import json
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import shutil
from tqdm import tqdm
from PIL import Image

import pydrake.all
from pydrake.all import RigidTransform, RollPitchYaw
# from manipulation.scenarios import ycb
from manipulation.utils import colorize_labels

debug = False
path = '/tmp/clutter_maskrcnn_data'
num_batches = 10
num_col = 6
num_row = 6
num_cameras = num_col * num_row // 2
num_images = num_batches * num_cameras

ycb = [
    "ball1/ball1.sdf", "ball2/ball2.sdf", "ball3/ball3.sdf",
    "bottle/bottle.sdf", "box1/box1.sdf", "box2/box2.sdf", "box3/box3.sdf",
    "can1/can1.sdf", "can2/can2.sdf", "can3/can3.sdf", "cup1/cup1.sdf", "cup2/cup2.sdf", "mug/mug.sdf", "plate/plate.sdf",

    # Balance out the distribution of items
    "bottle/bottle.sdf", "bottle/bottle.sdf",
    "mug/mug.sdf", "mug/mug.sdf",
    "plate/plate.sdf", "plate/plate.sdf"
]

name_to_class_id = dict()
for b in ["ball1/ball1", "ball2/ball2", "ball3/ball3"]:
    name_to_class_id[b] = 1
name_to_class_id["plate/plate"] = 2
name_to_class_id["mug/mug"] = 3
for b in ["box1/box1", "box2/box2", "box3/box3"]:
    name_to_class_id[b] = 4
name_to_class_id["bottle/bottle"] = 5
for c in ["cup1/cup1", "cup2/cup2"]:
    name_to_class_id[c] = 6
for c in ["can1/can1", "can2/can2", "can3/can3"]:
    name_to_class_id[c] = 7

instance_id_to_class_id = dict()


if not debug:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    print(f'Creating dataset in {path} with {num_images} images')

rng = np.random.default_rng()  # this is for python
generator = pydrake.common.RandomGenerator(rng.integers(1000))  # for c++

def generate_image(image_num):
    filename_base = os.path.join(path, f"{image_num:05d}")

    builder = pydrake.systems.framework.DiagramBuilder()
    plant, scene_graph = pydrake.multibody.plant.AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
    parser = pydrake.multibody.parsing.Parser(plant)
    parser.AddModelFromFile("ycb/table_surface.sdf")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("top_center"))
    inspector = scene_graph.model_inspector()

    instance_id_to_class_name = dict()

    for object_num in range(rng.integers(5,10)):
        this_object = ycb[rng.integers(len(ycb))]
        # this_object = ycb[2]
        class_name = os.path.splitext(this_object)[0]
        sdf = "ycb/" + this_object
        instance = parser.AddModelFromFile(sdf, f"object{object_num}")

        frame_id = plant.GetBodyFrameIdOrThrow(
            plant.GetBodyIndices(instance)[0])
        geometry_ids = inspector.GetGeometries(
            frame_id, pydrake.geometry.Role.kPerception)
        for geom_id in geometry_ids:
            instance_id_to_class_name[int(inspector.GetPerceptionProperties(geom_id).GetProperty("label", "id"))] = class_name
            instance_id_to_class_id[int(inspector.GetPerceptionProperties(geom_id).GetProperty("label", "id"))] = name_to_class_id[class_name]
    plant.Finalize()

    # print(instance_id_to_class_id)

    if not debug:
        # with open(filename_base + ".json", "w") as f:
        #     json.dump(instance_id_to_class_name, f)
        pass

    renderer = "my_renderer"
    scene_graph.AddRenderer(
        renderer, pydrake.geometry.render.MakeRenderEngineVtk(pydrake.geometry.render.RenderEngineVtkParams()))
    properties = pydrake.geometry.render.DepthCameraProperties(width=640,
                                        height=480,
                                        fov_y=np.pi / 5,
                                        renderer_name=renderer,
                                        z_near=0.1,
                                        z_far=10.0)

    for i in range(num_cameras):
        camera = builder.AddSystem(
            pydrake.systems.sensors.RgbdSensor(parent_id=scene_graph.world_frame_id(),
                        X_PB=gen_camera_pose(),
                        properties=properties,
                        show_window=False))
        camera.set_name("rgbd_sensor" + str(i))
        builder.Connect(scene_graph.get_query_output_port(),
                        camera.query_object_input_port())
        builder.ExportOutput(camera.color_image_output_port(), "color_image" + str(i))
        builder.ExportOutput(camera.label_image_output_port(), "label_image" + str(i))

    diagram = builder.Build()

    while True:
        simulator = pydrake.systems.analysis.Simulator(diagram)
        context = simulator.get_mutable_context()
        plant_context = plant.GetMyContextFromRoot(context)

        z = 0.05
        for body_index in plant.GetFloatingBaseBodies():
            tf = RigidTransform(
                    pydrake.math.UniformlyRandomRotationMatrix(generator),  
                    [rng.uniform(-.1, .1), rng.uniform(-.1, .1), z])
            plant.SetFreeBodyPose(plant_context, 
                                  plant.get_body(body_index),
                                  tf)
            z += 0.05

        try:
            simulator.AdvanceTo(2)
            break
        except RuntimeError:
            # I've chosen an aggressive simulation time step which works most 
            # of the time, but can fail occasionally.
            pass

    color_images = [diagram.GetOutputPort("color_image" + str(i)).Eval(context) for i in range(num_cameras)]
    label_images = [diagram.GetOutputPort("label_image" + str(i)).Eval(context) for i in range(num_cameras)]

    color_images_data = np.array([c.data for c in color_images])
    label_images_data = np.array([l.data for l in label_images])
    
    adj_label_images_data = np.zeros_like(label_images_data)
    
    for inst_id, class_id in instance_id_to_class_id.items():
        adj_label_images_data[label_images_data==inst_id] = class_id

    if debug: 
        # plt.figure()
        fig, axs = plt.subplots(num_row, num_col, sharex='col', sharey='row')
        for i in range(num_row):
            for j in range(num_col//2):
                idx = i * num_col//2 + j
                axs[i, 2*j].imshow(color_images_data[idx])
                axs[i, 2*j+1].imshow(colorize_labels(adj_label_images_data[idx]))
                axs[i, 2*j].axis('off')
                axs[i, 2*j+1].axis('off')
                
        plt.show()
    else:
        print(f"Saving batch {filename_base}")
        for idx in range(num_cameras):
            Image.fromarray(color_images_data[idx]).save(f"{filename_base}_{idx}.png")
            np.save(f"{filename_base}_{idx}_mask", adj_label_images_data[idx])

def gen_camera_pose():
    # Generate vector with length l, set as position of camera (x, y, z)
    # Calculate angle such that camera points into the center of the table (rpy)
    
    camera_pose = RigidTransform(RollPitchYaw(np.pi, 0, 0), [0, 0, 1])
    #  rng.uniform(-np.pi/2 , np.pi/2)
    camera_rotation = RigidTransform(RollPitchYaw(rng.uniform(np.pi/6, 5 * np.pi/12), 0,  rng.uniform(0, 2 * np.pi)), [0, 0, 0])
    final_pose = camera_rotation.multiply(camera_pose)
    return final_pose

if debug:
    for image_num in range(num_images):
        generate_image(image_num)
else:
    # pool = multiprocessing.Pool(10)
    # list(tqdm(pool.imap(generate_image, range(num_images)), total=num_images))
    # pool.close()
    # pool.join()
    for i in range(num_batches):
        generate_image(i)