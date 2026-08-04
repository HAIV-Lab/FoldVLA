from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# load external package
import os
import sys
import time
import numpy as np
import open3d as o3d
from termcolor import cprint
import threading

# load isaac-relevant package
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
from isaacsim.core.utils.types import ArticulationAction
from omni.physx.scripts import deformableUtils,particleUtils,physicsUtils



# load custom package
sys.path.append(os.getcwd())
from Env_StandAlone.BaseEnv import BaseEnv
from Env_Config.Garment.Particle_Garment import Particle_Garment
from Env_Config.Garment.Deformable_Garment import Deformable_Garment
from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e
from Env_Config.Camera.Recording_Camera import Recording_Camera
from Env_Config.Room.Real_Ground import Real_Ground
from Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from Env_Config.Utils_Project.Parse import parse_args_record
from Env_Config.Utils_Project.Flatten_Judge import judge_fling
from Env_Config.Room.Object_Tools import set_prim_visible_group, delete_prim_group
from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation
from Env_Config.Teleoperation.Listener import Listener


class TeleOp_Env(BaseEnv):
    def __init__(self):
        # load BaseEnv
        super().__init__()
        
        # ------------------------------------ #
        # ---        Add Env Assets        --- #
        # ------------------------------------ #
        self.ground = Real_Ground(
            self.scene, 
            visual_material_usd = None,
            # you can use materials in 'Assets/Material/Floor' to change the texture of ground.
        )        

        # load bimanual_dex
        self.bimanual_dex = Bimanual_Ur10e(
            self.world,
            dexleft_pos=np.array([-0.5, 0.0, 0.5]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.5, 0.0, 0.5]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )
        
        # ------------------------------------ #
        # --- Initialize World to be Ready --- #
        # ------------------------------------ #
        # initialize world
        self.reset()

        # step world to make it ready
        for i in range(100):
            self.step()
        
        cprint("World Ready!", "green", "on_green")
        
if __name__=="__main__":
    
    env = TeleOp_Env()
    
    env.listener = Listener(simulation_app, "handler")

    env.listener.launch()

    while simulation_app.is_running():

        env.step()

        hand_pose_rawR, arm_pose_rawR, hand_joint_poseR, wrist_posR, wrist_oriR = env.listener.get_pose("right")

        left_arm_pose = ArticulationAction(
            joint_positions=np.array([
                -1.57, -1.84, -2.5, -1.89, -1.57, 0.0
            ]),
            joint_indices=np.array(env.bimanual_dex.dexleft.arm_dof_indices)
        )
        
        right_arm_pose = ArticulationAction(
            joint_positions=np.array([
                1.57, -1.3, 2.5, -1.25, 1.57, 0.0
            ]),
            joint_indices=np.array(env.bimanual_dex.dexright.arm_dof_indices)
        )
        
        env.bimanual_dex.dexleft.apply_action(left_arm_pose)
        
        env.bimanual_dex.dexright.apply_action(right_arm_pose)

        if hand_joint_poseR is not None:
            env.bimanual_dex.dexright.set_joint_positions(hand_joint_poseR, env.bimanual_dex.dexright.hand_dof_indices)
            env.bimanual_dex.dexleft.set_joint_positions(hand_joint_poseR, env.bimanual_dex.dexright.hand_dof_indices)
            print(hand_joint_poseR)


    while simulation_app.is_running():
        simulation_app.update()
            

    
simulation_app.close()