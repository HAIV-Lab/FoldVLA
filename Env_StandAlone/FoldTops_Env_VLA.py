import os
import sys
import time
import numpy as np
import open3d as o3d
from termcolor import cprint
import threading
import omni.replicator.core as rep
import isaacsim.core.utils.prims as prims_utils
from pxr import UsdGeom,UsdPhysics,PhysxSchema, Gf
from isaacsim.core.api import World
from isaacsim.core.api import SimulationContext
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from isaacsim.core.utils.prims import is_prim_path_valid, set_prim_visibility
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from isaacsim.core.prims import SingleXFormPrim, SingleClothPrim, SingleRigidPrim, SingleGeometryPrim, SingleParticleSystem, SingleDeformablePrim
from isaacsim.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from omni.physx.scripts import deformableUtils,particleUtils,physicsUtils
sys.path.append(os.getcwd())
from Env_StandAlone.BaseEnv import BaseEnv
from Env_Config.Garment.Particle_Garment import Particle_Garment
from Env_Config.Garment.Deformable_Garment import Deformable_Garment
from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e
from Env_Config.Camera.Recording_Camera import Recording_Camera
from Env_Config.Room.Real_Ground import Real_Ground
from Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from Env_Config.Utils_Project.Parse import parse_args_record
from Env_Config.Utils_Project.Position_Judge import judge_pcd
from Env_Config.Room.Object_Tools import set_prim_visible_group, delete_prim_group
from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation

class FoldTops_Env(BaseEnv):
    def __init__(
        self, 
        pos:np.ndarray=None, 
        ori:np.ndarray=None, 
        usd_path:str=None, 
        ground_material_usd:str=None,
        record_video_flag:bool=False, 
    ):
        # load BaseEnv
        super().__init__()
        
        # ------------------------------------ #
        # ---        Add Env Assets        --- #
        # ------------------------------------ #
        self.ground = Real_Ground(
            self.scene, 
            visual_material_usd = ground_material_usd,
            # you can use materials in 'Assets/Material/Floor' to change the texture of ground.
        )
        
        # load garment
        self.garment = Particle_Garment(
            self.world, 
            pos=np.array([0, 3.0, 0.6]),
            ori=np.array([0.0, 0.0, 0.0]),
            usd_path=os.getcwd() + "/" + "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_018/TCLC_018_obj.usd" if usd_path is None else usd_path,
            contact_offset=0.012,             
            rest_offset=0.010,                
            particle_contact_offset=0.012,    
            fluid_rest_offset=0.010,
            solid_rest_offset=0.010,
        )
        # Here are some example garments you can try:
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Jacket032/TCLC_Jacket032_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Jacket152/TCLC_Jacket152_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top566/TCLC_Top566_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top584/TCLC_Top584_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_top118/TCLC_top118_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top476/TCLC_Top476_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top030/TCLC_Top030_obj.usd",  

        # load bimanual_dex
        self.bimanual_dex = Bimanual_Ur10e(
            self.world,
            dexleft_pos=np.array([-0.8, 0.0, 0.5]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.8, 0.0, 0.5]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )

        # load camera
        self.garment_camera = Recording_Camera(
            camera_position=np.array([0.0, 1.0, 6.75]), 
            camera_orientation=np.array([0, 90.0, 90.0]),
            prim_path="/World/garment_camera",
        )
        
        self.env_camera = Recording_Camera(
            camera_position=np.array([0.0, 4.0, 6.0]),
            camera_orientation=np.array([0, 60, -90.0]),
            prim_path="/World/env_camera",
        )
        
        self.garment_pcd = None
        self.points_affordance_feature = None
        
        # load Galbot robot (display only)
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import SingleXFormPrim
        add_reference_to_stage(usd_path="/new_data/why/isaac/galbot_one_golf_description/usd/galbot_one_golf.usda", prim_path="/World/Galbot")
        self.galbot = SingleXFormPrim(
            prim_path="/World/Galbot",
            name="galbot",
            position=np.array([-1.2, 0.8, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        # load SharpaWave hands (display only)
        add_reference_to_stage(usd_path="/new_data/why/isaac/sharpa-urdf-usd-xml/wave_01/left_sharpa_wave/left_sharpa_wave_with_wrist.usda", prim_path="/World/SharpaLeft")
        self.sharpa_left = SingleXFormPrim(
            prim_path="/World/SharpaLeft",
            name="sharpa_left",
            position=np.array([0.0, 0.8, 0.5]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        add_reference_to_stage(usd_path="/new_data/why/isaac/sharpa-urdf-usd-xml/wave_01/right_sharpa_wave/right_sharpa_wave_with_wrist.usda", prim_path="/World/SharpaRight")
        self.sharpa_right = SingleXFormPrim(
            prim_path="/World/SharpaRight",
            name="sharpa_right",
            position=np.array([0.0, 0.8, 0.5]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        # load GAM Model
        self.model = GAM_Encapsulation(catogory="Tops_LongSleeve")   
        
        # ------------------------------------ #
        # --- Initialize World to be Ready --- #
        # ------------------------------------ #
        # initialize world
        self.reset()
        
        # 把左右手分别对齐到 Galbot 手腕
        from isaacsim.core.prims import SingleXFormPrim as XFormPrim
        left_wrist = XFormPrim("/World/Galbot/left_arm_link7/left_arm_end_effector_mount_link")
        right_wrist = XFormPrim("/World/Galbot/right_arm_link7/right_arm_end_effector_mount_link")
        left_pos, left_ori = left_wrist.get_world_pose()
        right_pos, right_ori = right_wrist.get_world_pose()
        from isaacsim.core.utils.rotations import euler_angles_to_quat
        left_fix_ori = euler_angles_to_quat(np.array([-90, 0, 0]), degrees=True)
        right_fix_ori = euler_angles_to_quat(np.array([90, 0, 0]), degrees=True)
        left_pos_adjusted = left_pos + np.array([-0.005, 0.0, 0.0])
        self.sharpa_left.set_world_pose(position=left_pos_adjusted, orientation=left_fix_ori)
        self.sharpa_right.set_world_pose(position=right_pos, orientation=right_fix_ori)
        print(f"左手位置: {left_pos}")
        print(f"右手位置: {right_pos}")
        # move garment to the target position
        self.garment.set_pose(pos=np.array([pos[0], pos[1], 0.2]), ori=ori)
        self.position = [pos[0], pos[1], 0.2]
        self.orientation = ori
        
        self.garment_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Garment/garment"
            ]
        )
        
        self.env_camera.initialize(depth_enable=True)
        
        # add thread and record gif Asynchronously(use to collect rgb data for generating gif)
        if record_video_flag:
            self.thread_record = threading.Thread(target=self.env_camera.collect_rgb_graph_for_video)
            self.thread_record.daemon = True
        
        # open hand to be initial state
        self.bimanual_dex.set_both_hand_state("open", "open")

        # step world to make it ready
        for i in range(100):
            self.step()
            
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        cprint(f"usd_path: {usd_path}", "magenta")
        cprint(f"pos_x: {pos[0]}", "magenta")
        cprint(f"pos_y: {pos[1]}", "magenta")
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        
        cprint("World Ready!", "green", "on_green")
    
    def record_callback(self, step_size):

        if self.step_num % 5 == 0:
        
            joint_pos_L = self.bimanual_dex.dexleft.get_joint_positions()
            
            joint_pos_R = self.bimanual_dex.dexright.get_joint_positions()
            
            joint_state = np.array([*joint_pos_L, *joint_pos_R])

            rgb = self.env_camera.get_rgb_graph(save_or_not=False)

            point_cloud = self.env_camera.get_pointcloud_from_depth(
                show_original_pc_online=False,
                show_downsample_pc_online=False,
            )
            
            self.saving_data.append({ 
                "joint_state": joint_state,
                "image": rgb,
                "env_point_cloud": point_cloud,
                "garment_point_cloud":self.garment_pcd,
                "points_affordance_feature": self.points_affordance_feature,
            })
        
        self.step_num += 1


