# MyPegInsertion/base.py
"""Base class for MyPegInsertion environment."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.MyPegInsertion import constants as consts


def get_assets() -> Dict[str, bytes]:
    """Returns a dictionary of all assets used by the environment."""
    assets = {}
    # 加载你的 assets 目录
    path = epath.Path(__file__).parent / "xmls" / "assets"
    if path.exists():
        mjx_env.update_assets(assets, path)
    return assets


class MyPegInsertionEnv(mjx_env.MjxEnv):
    """Base class for MyPegInsertion environment."""

    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        # 加载资产并创建模型（与 ALOHA 方式一致）
        self._model_assets = get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(
            epath.Path(xml_path).read_text(), assets=self._model_assets
        )
        self._mj_model.opt.timestep = self._config.sim_dt

        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = xml_path

    def _post_init(self, keyframe: str = "home"):
        """初始化机器人属性（根据你的 XML 中的名称）"""
     # 末端执行器 site（使用你的 XML 中的名称）
        self._left_gripper_site = self._mj_model.site("left_pinch").id
        self._right_gripper_site = self._mj_model.site("right_pinch").id

     # 初始关节位置和控制信号
        self._init_q = jp.array(self._mj_model.keyframe(keyframe).qpos)
        self._init_ctrl = jp.array(self._mj_model.keyframe(keyframe).ctrl)

     # 执行器控制范围
        self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T
 
     # 手臂关节的 qpos 地址
        arm_joint_ids = [self._mj_model.joint(j).id for j in consts.ARM_JOINTS]
        self._arm_qadr = jp.array(
            [self._mj_model.jnt_qposadr[joint_id] for joint_id in arm_joint_ids]
        )

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
