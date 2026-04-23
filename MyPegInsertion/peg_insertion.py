# MyPegInsertion/peg_insertion.py
"""Peg insertion task for MyPegInsertion environment."""

from typing import Any, Dict, Optional, Union

import jax
from jax import numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.MyPegInsertion import base
from mujoco_playground._src.manipulation.MyPegInsertion import constants as consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.0025,
        sim_dt=0.0025,
        episode_length=1000,
        action_repeat=2,
        action_scale=0.005,
        reward_config=config_dict.create(
            scales=config_dict.create(
                distance=1.0,
                success=10.0,
            )
        ),
        success_threshold=0.02,
        impl="jax",
        naconmax=24 * 1024,
        njmax=256,
    )


class MyPegInsertion(base.MyPegInsertionEnv):
    """Peg insertion task for dual-arm robot with two pegs and two holes."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        # XML 文件路径
        xml_path = "/home/yr/桌面/aloha_self/mujoco_playground/mujoco_playground/_src/manipulation/MyPegInsertion/xmls/my_peg_insertion.xml"
        super().__init__(xml_path=xml_path, config=config, config_overrides=config_overrides)
        self._post_init()
        self._success_threshold = config.get("success_threshold", 0.02)

    def _post_init(self):
        """初始化：获取模型中的 body 和 site ID"""
        super()._post_init()

        # 获取 peg 和 hole 的 body ID
        self._peg_body = self._mj_model.body("truss_frame2").id
        self._hole_body = self._mj_model.body("truss_frame1").id

        # 获取 peg tip site ID（两个）
        self._peg_tip1_site = self._mj_model.site("peg_tip1").id
        self._peg_tip2_site = self._mj_model.site("peg_tip2").id

        # 获取 hole entrance site ID（两个）
        self._hole_entrance1_site = self._mj_model.site("hole_entrance1").id
        self._hole_entrance2_site = self._mj_model.site("hole_entrance2").id

        # 获取 hole bottom site ID（两个，用于深度奖励，可选）
        self._hole_bottom1_site = self._mj_model.site("hole_bottom1").id
        self._hole_bottom2_site = self._mj_model.site("hole_bottom2").id

        # 获取 peg 和 hole 的 qpos 地址（用于 reset 时随机化位置）
        self._peg_qadr = self._mj_model.jnt_qposadr[
            self._mj_model.body_jntadr[self._peg_body]
        ]
        self._hole_qadr = self._mj_model.jnt_qposadr[
            self._mj_model.body_jntadr[self._hole_body]
        ]

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """重置环境，随机化 peg 和 hole 的初始 xy 位置"""
        rng, rng_peg, rng_hole = jax.random.split(rng, 3)

        # 随机化 peg 和 hole 的 xy 位置（范围 ±0.1）
        peg_xy = jax.random.uniform(rng_peg, (2,), minval=-0.1, maxval=0.1)
        hole_xy = jax.random.uniform(rng_hole, (2,), minval=-0.1, maxval=0.1)

        init_q = self._init_q.at[self._peg_qadr : self._peg_qadr + 2].add(peg_xy)
        init_q = init_q.at[self._hole_qadr : self._hole_qadr + 2].add(hole_xy)

        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=self._init_ctrl,
            impl=self._mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )

        info = {"rng": rng}
        obs = self._get_obs(data)
        reward, done = jp.zeros(2)
        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=float),
            "dist_left": jp.array(0.0, dtype=float),
            "dist_right": jp.array(0.0, dtype=float),
        }

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """执行一步仿真"""
        delta = action * self._config.action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        # 检查是否飞出边界
        out_of_bounds = jp.any(jp.abs(data.xpos[self._peg_body]) > 1.0)
        out_of_bounds |= jp.any(jp.abs(data.xpos[self._hole_body]) > 1.0)

        # 计算奖励
        raw_rewards = self._get_reward(data)
        rewards = {
            k: v * self._config.reward_config.scales[k]
            for k, v in raw_rewards.items()
        }
        reward = sum(rewards.values()) / sum(
            self._config.reward_config.scales.values()
        )

        done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)

        # 更新 metrics
        state.metrics.update(
            **rewards,
            out_of_bounds=out_of_bounds.astype(float),
            dist_left=raw_rewards["dist_left"],
            dist_right=raw_rewards["dist_right"],
        )

        obs = self._get_obs(data)
        return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        """构建观察向量"""
        # 左臂末端位置
        left_gripper_pos = data.site_xpos[self._left_gripper_site]
        # 右臂末端位置
        right_gripper_pos = data.site_xpos[self._right_gripper_site]

        # peg tip 位置
        peg_tip1_pos = data.site_xpos[self._peg_tip1_site]
        peg_tip2_pos = data.site_xpos[self._peg_tip2_site]

        # hole entrance 位置
        hole_entrance1_pos = data.site_xpos[self._hole_entrance1_site]
        hole_entrance2_pos = data.site_xpos[self._hole_entrance2_site]

        # 相对位置
        peg_to_hole1 = hole_entrance1_pos - peg_tip1_pos
        peg_to_hole2 = hole_entrance2_pos - peg_tip2_pos

        obs = jp.concatenate([
            data.qpos,                      # 所有关节位置
            data.qvel,                      # 所有关节速度
            left_gripper_pos,               # 左臂末端 (3)
            right_gripper_pos,              # 右臂末端 (3)
            peg_tip1_pos,                   # peg tip1 位置 (3)
            peg_tip2_pos,                   # peg tip2 位置 (3)
            hole_entrance1_pos,             # hole entrance1 位置 (3)
            hole_entrance2_pos,             # hole entrance2 位置 (3)
            peg_to_hole1,                   # 相对位置1 (3)
            peg_to_hole2,                   # 相对位置2 (3)
        ])

        return obs

    def _get_reward(self, data: mjx.Data) -> Dict[str, jax.Array]:
        """计算奖励：距离奖励 + 成功奖励"""
        # 获取位置
        peg_tip1_pos = data.site_xpos[self._peg_tip1_site]
        peg_tip2_pos = data.site_xpos[self._peg_tip2_site]
        hole_entrance1_pos = data.site_xpos[self._hole_entrance1_site]
        hole_entrance2_pos = data.site_xpos[self._hole_entrance2_site]

        # 计算两个 peg 到对应 hole entrance 的距离
        dist_left = jp.linalg.norm(peg_tip1_pos - hole_entrance1_pos)
        dist_right = jp.linalg.norm(peg_tip2_pos - hole_entrance2_pos)

        # 总距离
        total_distance = dist_left + dist_right

        # 距离奖励（负值，越小越好）
        distance_reward = -total_distance

        # 成功奖励（两个 peg 都插入成功）
        success = (dist_left < self._success_threshold) & (dist_right < self._success_threshold)
        success_reward = success.astype(float) * 10.0

        return {
            "distance": distance_reward,
            "success": success_reward,
            "dist_left": dist_left,
            "dist_right": dist_right,
        }