# -*- coding: utf-8 -*-

"""Probe RZ/XY phase-aware SQNN variants on MaxCut-3.

这个文件是当前 V14-Z-Phase 主线的核心实验脚本。它做三件事：

1. 定义 ``PhaseAwareJRegularizedSQNN``，也就是每个变量一个 Bloch 向量
   ``(X, Y, Z)`` 的相位感知 SQNN。
2. 构造一组 ``phase_mode`` 变体，例如 memory-XY、neighbor-XY、
   Z-edge cavity message、late collapse 和 final rotation。
3. 训练每个变体，并把每轮概率、direct rounding、sampling、local search
   的质量写到 ``summary.csv`` / ``metrics.json``。

重要指标提醒：
当前 MaxCut-3 生成器把 ``benchmark.known_optimum`` 设成总边权 ``W``，
不是精确最优 cut ``C*``。因此本脚本里历史字段名为 ``*_ratio`` 的
MaxCut-3 数值，默认实际含义是 ``C/W`` cut fraction；只有当调用方显式
传入精确 ``C*`` 或 best-known cut 时，它才是 ``C/C*`` 或
``C/C_best_known``。
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_j_regularized_sqnn import (  # noqa: E402
    SUMMARY_FIELDS,
    config_id,
    evaluate_solution_quality,
    j_penalty_value,
    load_summary,
    make_train_args,
    make_warm_start_probabilities,
)
from quantum.core.layers import _apply_bloch_noise, _apply_bloch_rotation  # noqa: E402
from quantum.warmstart.losses import bernoulli_entropy  # noqa: E402
from quantum.warmstart.qubo import QUBOProblem  # noqa: E402
from quantum.warmstart.qubo_sqnn import bloch_to_probabilities  # noqa: E402
from run_qubo_warmstart import make_benchmark  # noqa: E402


BASE_RUN_ID = "maxcut3_learn_strength_chase_random_regular_maxcut_n512_d3p0_s42_jw100p0_relu_25e1e7ec86"

EXTRA_SUMMARY_FIELDS = [
    "phase_mode",
    "phase_memory_decay",
    "xy_feedback_init",
    "xy_feedback_active_fraction",
    "xy_feedback_decay_fraction",
    "omega_init",
    "neighbor_phase_init",
    "phase_diff_init",
    "collapse_init",
    "final_rotation_max",
    "edge_message_decay",
    "edge_message_self_mix",
    "z_message_decay",
    "z_message_self_mix",
    "z_message_gain",
    "z_message_gain_final",
    "z_message_gain_schedule_start",
    "head_count",
    "head_seed_stride",
    "node_step_mode",
    "rollback_aux_on_reject",
    "vector_loss_weight",
    "vector_best_ratio",
    "vector_final_ratio",
    "final_xy_radius",
    "final_rotation_norm",
]
PHASE_SUMMARY_FIELDS = list(dict.fromkeys([*SUMMARY_FIELDS, *EXTRA_SUMMARY_FIELDS]))


class PhaseAwareJRegularizedSQNN(nn.Module):
    """V14 相位感知 SQNN 主模型。

    阅读这段代码时，可以按下面的物理/算法含义对应：

    - 一个 QUBO/MaxCut 变量 i 对应一个 Bloch 向量 ``r_i=(X_i,Y_i,Z_i)``。
    - 最终只用 Z 基读出概率：``p_i=P(x_i=1)=(1-Z_i)/2``，所以训练主目标
      仍是 Z-basis/product-distribution MaxCut。
    - RZ 只在 X/Y 平面里转相位，不直接改变 ``p_i``。
    - RY 把 X/Y 相位信息折回 Z，从而真正改变 ``p_i``。
    - ``J_i=-F_i*(p_i^{new}-p_i^{old})`` 用来约束更新方向不要明显背离局部
      能量下降方向。
    """

    def __init__(
        self,
        num_variables,
        message_rounds,
        noise_config=None,
        step_init=0.25,
        phase_init=0.10,
        mixer_bias_init=0.0,
        monotone_accept=True,
        normalize_local_field=True,
        trust_mode="fixed",
        trust_shrink=1.0,
        trust_threshold=0.0,
        adaptive_trust_min=0.0,
        adaptive_trust_scale=1e-3,
        two_stage_fraction=0.0,
        symmetry_breaking="none",
        symmetry_strength=0.0,
        symmetry_strength_trainable=False,
        symmetry_strength_max=0.5,
        symmetry_seed=0,
        initial_probabilities=None,
        phase_mode="baseline",
        phase_memory_decay=0.0,
        xy_feedback_init=0.0,
        xy_feedback_active_fraction=1.0,
        xy_feedback_decay_fraction=0.0,
        omega_init=0.0,
        neighbor_phase_init=0.0,
        phase_diff_init=0.0,
        collapse_init=0.0,
        final_rotation_max=0.0,
        edge_message_decay=0.70,
        edge_message_self_mix=0.50,
        z_message_decay=0.70,
        z_message_self_mix=0.50,
        z_message_gain=1.0,
        z_message_gain_final=None,
        z_message_gain_schedule_start=0.60,
        node_step_mode="none",
        rollback_aux_on_reject=False,
    ):
        super().__init__()
        # ===== 1. 保存结构和控制开关 =====
        # num_variables 是图上变量/节点数；message_rounds 是 SQNN 展开轮数。
        self.num_variables = int(num_variables)
        self.message_rounds = int(message_rounds)
        self.noise_config = noise_config
        # monotone_accept=True 时，一整轮 proposal 只有在期望能量不升时才接受。
        self.monotone_accept = bool(monotone_accept)
        # normalize_local_field=True 会把局部场按节点强度归一化，避免高度节点步长过大。
        self.normalize_local_field = bool(normalize_local_field)
        # trust_* 是局部 trust region，用 J<0 识别“方向可疑”的节点更新并缩小步长。
        self.trust_mode = str(trust_mode)
        self.trust_shrink = float(trust_shrink)
        self.trust_threshold = float(trust_threshold)
        self.adaptive_trust_min = float(adaptive_trust_min)
        self.adaptive_trust_scale = float(adaptive_trust_scale)
        # two_stage_fraction 用于“前期探索、后期 collapse/trust”的两阶段策略。
        self.two_stage_fraction = float(two_stage_fraction)
        # symmetry_breaking 给所有节点一个很小的初始差异，否则正则图容易全体对称。
        self.symmetry_breaking = str(symmetry_breaking)
        self.symmetry_strength_trainable = bool(symmetry_strength_trainable)
        self.symmetry_strength_max = float(symmetry_strength_max)
        self.symmetry_seed = int(symmetry_seed)
        # phase_mode 是实验开关字符串，例如 memory_xy_feedback_z_edge_cavity_collapse。
        self.phase_mode = str(phase_mode)
        self.phase_memory_decay = float(phase_memory_decay)
        self.xy_feedback_active_fraction = float(xy_feedback_active_fraction)
        self.xy_feedback_decay_fraction = float(xy_feedback_decay_fraction)
        self.final_rotation_max = float(final_rotation_max)
        # edge_message_* 控制 XY cavity message 的平滑和自混合。
        self.edge_message_decay = float(edge_message_decay)
        self.edge_message_self_mix = float(edge_message_self_mix)
        # z_message_* 控制 MaxCut 专用的 Z-consistent edge/cavity message。
        self.z_message_decay = float(z_message_decay)
        self.z_message_self_mix = float(z_message_self_mix)
        self.z_message_gain = float(z_message_gain)
        self.z_message_gain_final = (
            None if z_message_gain_final is None else float(z_message_gain_final)
        )
        self.z_message_gain_schedule_start = float(z_message_gain_schedule_start)
        self.node_step_mode = str(node_step_mode)
        self.rollback_aux_on_reject = bool(rollback_aux_on_reject)

        # ===== 2. 初始 warm-start 概率 =====
        # 如果给了 initial_probabilities，就把它编码成初始 Z；否则从 |+> / X 正方向开始。
        if initial_probabilities is None:
            initial_probabilities = torch.empty(0)
        self.register_buffer(
            "initial_probabilities",
            torch.as_tensor(initial_probabilities, dtype=torch.get_default_dtype()).detach().clone(),
            persistent=False,
        )
        # ===== 3. 每一轮可训练角度/步长 =====
        # field_steps 控制 RY 局部场步长；phase_steps 控制 RZ 相位记忆强度。
        self.field_steps = nn.Parameter(torch.full((self.message_rounds,), float(step_init)))
        self.phase_steps = nn.Parameter(torch.full((self.message_rounds,), float(phase_init)))
        # mixer_bias 是每轮统一的 RY 偏置。
        self.mixer_bias = nn.Parameter(torch.full((self.message_rounds,), float(mixer_bias_init)))
        # omega/double RZ、XY feedback、neighbor phase、phase diff、collapse 都是不同 phase_mode 的组件。
        self.omega_steps = nn.Parameter(torch.full((self.message_rounds,), float(omega_init)))
        self.xy_feedback_steps = nn.Parameter(torch.full((self.message_rounds,), float(xy_feedback_init)))
        self.neighbor_phase_steps = nn.Parameter(
            torch.full((self.message_rounds,), float(neighbor_phase_init))
        )
        self.phase_diff_steps = nn.Parameter(torch.full((self.message_rounds,), float(phase_diff_init)))
        self.collapse_steps = nn.Parameter(torch.full((self.message_rounds,), float(collapse_init)))
        self.final_rotation_raw = nn.Parameter(torch.zeros(3))
        self.initial_angles = nn.Parameter(torch.zeros(3))

        # node_step_mode="learned_gate" 时，这三个参数学习每个节点的有效 RY 步长。
        self.node_gate_bias = nn.Parameter(torch.tensor(0.0))
        self.node_gate_field = nn.Parameter(torch.tensor(1.0))
        self.node_gate_confidence = nn.Parameter(torch.tensor(-0.5))

        # ===== 4. 破对称强度：可固定，也可训练 =====
        initial_strength = float(symmetry_strength)
        if self.symmetry_strength_trainable:
            max_strength = max(float(self.symmetry_strength_max), 1e-6)
            clipped = min(max(initial_strength, 1e-6), max_strength - 1e-6)
            probability = clipped / max_strength
            raw = math.log(probability / max(1.0 - probability, 1e-12))
            self.raw_symmetry_strength = nn.Parameter(torch.tensor(float(raw)))
        else:
            self.register_buffer("fixed_symmetry_strength", torch.tensor(initial_strength))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _prepare_problem(self, problem):
        """把 QUBOProblem 移到模型所在 device/dtype，并做类型检查。"""
        if not isinstance(problem, QUBOProblem):
            raise TypeError("PhaseAwareJRegularizedSQNN expects a QUBOProblem")
        return problem.to(device=self.device, dtype=self.dtype)

    def current_symmetry_strength(self):
        """返回当前破对称强度；可训练版本用 sigmoid 限制在 [0, max]。"""
        if self.symmetry_strength_trainable:
            max_strength = torch.as_tensor(
                max(float(self.symmetry_strength_max), 1e-6),
                dtype=self.dtype,
                device=self.device,
            )
            return max_strength * torch.sigmoid(self.raw_symmetry_strength.to(device=self.device, dtype=self.dtype))
        return self.fixed_symmetry_strength.to(device=self.device, dtype=self.dtype)

    def _initial_bloch(self, problem):
        """构造初始 Bloch 状态。

        这里是模型的“初态”：
        - 有 warm-start 概率时，当前实现把概率 p 写成 Z=2p-1，同时给 X 留出
          合法半径；注意读出函数使用 p=(1-Z)/2，所以这里的 warm-start 极性
          是一个需要重点复核的历史实现点；
        - 没有 warm-start 时，从 X=1, Y=0, Z=0 开始，也就是 p=0.5；
        - symmetry_breaking 会对 RY/RZ 加小扰动，打破正则图中的节点对称性。
        """
        bloch = torch.zeros((problem.num_variables, 3), dtype=self.dtype, device=self.device)
        if self.initial_probabilities.numel() == problem.num_variables:
            initial = self.initial_probabilities.to(device=self.device, dtype=self.dtype).clamp(1e-6, 1.0 - 1e-6)
            z_value = 2.0 * initial - 1.0
            bloch[:, 0] = torch.sqrt((1.0 - z_value * z_value).clamp_min(0.0))
            bloch[:, 2] = z_value
        else:
            bloch[:, 0] = 1.0

        angles = self.initial_angles.to(dtype=self.dtype, device=self.device).expand(
            problem.num_variables,
            -1,
        ).clone()
        strength = self.current_symmetry_strength()
        if self.symmetry_breaking != "none" and bool((strength > 0.0).detach().item()):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.symmetry_seed)
            mode = "random_ry" if self.symmetry_breaking == "random_z" else self.symmetry_breaking
            if mode in {"random_ry", "random_rz", "random_rz_ry"}:
                noise = 2.0 * torch.rand(problem.num_variables, generator=gen) - 1.0
                noise = noise.to(device=self.device, dtype=self.dtype)
                if mode in {"random_ry", "random_rz_ry"}:
                    angles[:, 1] = angles[:, 1] + strength * noise
                if mode in {"random_rz", "random_rz_ry"}:
                    rz_noise = 2.0 * torch.rand(problem.num_variables, generator=gen) - 1.0
                    rz_noise = rz_noise.to(device=self.device, dtype=self.dtype)
                    angles[:, 0] = angles[:, 0] + strength * rz_noise
            elif mode == "degree_hash":
                degree = problem.node_degrees(weighted=False).to(device=self.device, dtype=self.dtype)
                centered = degree - degree.mean()
                normalized = centered / centered.abs().max().clamp_min(1.0)
                angles[:, 1] = angles[:, 1] + strength * normalized
            else:
                raise ValueError(f"unknown symmetry_breaking: {self.symmetry_breaking}")
        return _apply_bloch_rotation(bloch, angles)

    def _probabilities_from_bloch(self, bloch):
        """从 Bloch 向量取 Z 基概率 p_i=P(x_i=1)。"""
        return bloch_to_probabilities(bloch)[:, 2]

    def _safe_project_bloch_ball(self, bloch):
        """数值保护：把长度超过 1 的 Bloch 向量投回 Bloch 球内。"""
        norm = torch.linalg.vector_norm(bloch, dim=-1, keepdim=True)
        return bloch / norm.clamp_min(1.0)

    def _local_field(self, problem, probabilities):
        """计算当前概率下每个变量的 QUBO 局部场 F_i。

        QUBO 期望能量对 p_i 的局部线性项近似为：

            F_i = linear_i + sum_j Q_ij p_j

        对 MaxCut，能量是 E_QUBO=-C。后面使用
        ``J_i=-F_i*(p_i_new-p_i_old)`` 判断这轮概率变化是否大体朝着降低
        QUBO 能量的方向走。
        """
        field = problem.linear.to(device=self.device, dtype=self.dtype).clone()
        if problem.edge_index.numel():
            src, dst = problem.edge_index
            edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
            field.index_add_(0, src, edge_weight * probabilities[dst])
            field.index_add_(0, dst, edge_weight * probabilities[src])

        if not self.normalize_local_field:
            return field

        normalizer = problem.linear.abs().to(device=self.device, dtype=self.dtype)
        normalizer = normalizer + problem.node_degrees(weighted=True, absolute=True).to(
            device=self.device,
            dtype=self.dtype,
        )
        return field / normalizer.clamp_min(1e-6)

    def _neighbor_xy_signal(self, problem, bloch):
        """聚合邻居的 X/Y 相位，得到一个简单的邻居相位力矩。

        这个信号只看节点的直接邻居，不排除反向回流。它用于早期 phase-aware
        对照：邻居相位是否能帮助最终 Z 概率形成更好的 cut。
        """
        if problem.edge_index.numel() == 0:
            zeros = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
            return zeros, zeros
        src, dst = problem.edge_index
        edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
        msg_x = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        msg_y = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        msg_x.index_add_(0, src, edge_weight * bloch[dst, 0])
        msg_x.index_add_(0, dst, edge_weight * bloch[src, 0])
        msg_y.index_add_(0, src, edge_weight * bloch[dst, 1])
        msg_y.index_add_(0, dst, edge_weight * bloch[src, 1])
        degree = problem.node_degrees(weighted=True, absolute=True).to(device=self.device, dtype=self.dtype)
        msg_x = msg_x / degree.clamp_min(1e-6)
        msg_y = msg_y / degree.clamp_min(1e-6)
        # torque is positive when the node XY phase trails its weighted
        # neighbor phase. Its trainable sign decides whether the model moves
        # toward or away from neighbor phase alignment.
        torque = bloch[:, 0] * msg_y - bloch[:, 1] * msg_x
        alignment = bloch[:, 0] * msg_x + bloch[:, 1] * msg_y
        return torque, alignment

    def _phase_difference_signal(self, problem, bloch):
        """用边两端的 XY 相位差 sin(phi_j-phi_i) 作为相位关系信号。"""
        if problem.edge_index.numel() == 0:
            return torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        src, dst = problem.edge_index
        edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
        phase = torch.atan2(bloch[:, 1], bloch[:, 0])
        edge_delta = torch.sin(phase[dst] - phase[src])
        signal = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        signal.index_add_(0, src, edge_weight * edge_delta)
        signal.index_add_(0, dst, -edge_weight * edge_delta)
        degree = problem.node_degrees(weighted=True, absolute=True).to(device=self.device, dtype=self.dtype)
        return signal / degree.clamp_min(1e-6)

    def _cavity_xy_signal(self, problem, bloch):
        """节点级 cavity XY 信号。

        cavity 的意思是：计算 i 接收邻居群体影响时，尽量扣掉当前这条边的
        直接回流。它比普通 neighbor_xy 更像消息传递中的 non-backtracking
        更新，但这里仍然先聚合到节点级。
        """
        if problem.edge_index.numel() == 0:
            zeros = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
            return zeros, zeros
        src, dst = problem.edge_index
        edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
        sum_x = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        sum_y = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        sum_x.index_add_(0, src, edge_weight * bloch[dst, 0])
        sum_x.index_add_(0, dst, edge_weight * bloch[src, 0])
        sum_y.index_add_(0, src, edge_weight * bloch[dst, 1])
        sum_y.index_add_(0, dst, edge_weight * bloch[src, 1])
        degree = problem.node_degrees(weighted=True, absolute=True).to(device=self.device, dtype=self.dtype)
        cavity_degree = (degree - 1.0).clamp_min(1.0)

        src_to_dst_x = (sum_x[src] - bloch[dst, 0]) / cavity_degree[src]
        src_to_dst_y = (sum_y[src] - bloch[dst, 1]) / cavity_degree[src]
        dst_to_src_x = (sum_x[dst] - bloch[src, 0]) / cavity_degree[dst]
        dst_to_src_y = (sum_y[dst] - bloch[src, 1]) / cavity_degree[dst]

        cav_x = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        cav_y = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        cav_x.index_add_(0, dst, edge_weight * src_to_dst_x)
        cav_y.index_add_(0, dst, edge_weight * src_to_dst_y)
        cav_x.index_add_(0, src, edge_weight * dst_to_src_x)
        cav_y.index_add_(0, src, edge_weight * dst_to_src_y)
        cav_x = cav_x / degree.clamp_min(1e-6)
        cav_y = cav_y / degree.clamp_min(1e-6)
        torque = bloch[:, 0] * cav_y - bloch[:, 1] * cav_x
        alignment = bloch[:, 0] * cav_x + bloch[:, 1] * cav_y
        return torque, alignment

    def _edge_cavity_xy_signal(self, problem, bloch, edge_message):
        """有向边级 XY cavity message。

        每条无向边 (i,j) 拆成 i->j 和 j->i 两条有向消息。更新 i->j 时，
        会从 i 收到的 incoming message 中排除 j->i 的直接回流，这就是
        cavity / non-backtracking 的部分。输出再聚合回节点，形成 RZ/RY
        可用的 XY torque。
        """
        if problem.edge_index.numel() == 0:
            zeros = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
            return zeros, zeros, edge_message

        src, dst = problem.edge_index
        edge_count = int(src.numel())
        tail = torch.cat((src, dst), dim=0)
        head = torch.cat((dst, src), dim=0)
        reverse = torch.cat(
            (
                torch.arange(edge_count, 2 * edge_count, device=self.device),
                torch.arange(0, edge_count, device=self.device),
            ),
            dim=0,
        )
        edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype)
        directed_weight = torch.cat((edge_weight, edge_weight), dim=0)

        if edge_message.numel() != 2 * edge_count * 2:
            edge_message = bloch[tail, :2]
        else:
            edge_message = edge_message.to(device=self.device, dtype=self.dtype)

        incoming = torch.zeros((problem.num_variables, 2), dtype=self.dtype, device=self.device)
        incoming.index_add_(0, head, directed_weight.unsqueeze(-1) * edge_message)

        degree = problem.node_degrees(weighted=True, absolute=True).to(device=self.device, dtype=self.dtype)
        cavity_degree = (degree[tail] - directed_weight).clamp_min(1.0)
        cavity = (incoming[tail] - directed_weight.unsqueeze(-1) * edge_message[reverse]) / cavity_degree.unsqueeze(-1)

        self_mix = torch.as_tensor(
            min(max(float(self.edge_message_self_mix), 0.0), 1.0),
            dtype=self.dtype,
            device=self.device,
        )
        raw_message = self_mix * bloch[tail, :2] + (1.0 - self_mix) * cavity
        raw_norm = torch.linalg.vector_norm(raw_message, dim=-1, keepdim=True)
        raw_message = raw_message / raw_norm.clamp_min(1.0)

        decay = torch.as_tensor(
            min(max(float(self.edge_message_decay), 0.0), 1.0),
            dtype=self.dtype,
            device=self.device,
        )
        next_message = decay * edge_message + (1.0 - decay) * raw_message
        message_norm = torch.linalg.vector_norm(next_message, dim=-1, keepdim=True)
        next_message = next_message / message_norm.clamp_min(1.0)

        node_message = torch.zeros((problem.num_variables, 2), dtype=self.dtype, device=self.device)
        node_message.index_add_(0, head, directed_weight.unsqueeze(-1) * next_message)
        node_message = node_message / degree.clamp_min(1e-6).unsqueeze(-1)
        torque = bloch[:, 0] * node_message[:, 1] - bloch[:, 1] * node_message[:, 0]
        alignment = bloch[:, 0] * node_message[:, 0] + bloch[:, 1] * node_message[:, 1]
        return torque, alignment, next_message

    def _current_z_message_gain(self, round_index):
        """返回当前轮的 Z-edge gain。

        如果没有设置 z_message_gain_final，就一直用固定 gain；否则从
        z_message_gain_schedule_start 对应轮次开始，线性插值到 final gain。
        """
        if self.z_message_gain_final is None:
            return float(self.z_message_gain)
        start_fraction = min(max(float(self.z_message_gain_schedule_start), 0.0), 1.0)
        start_round = int(round(float(self.message_rounds) * start_fraction))
        if round_index <= start_round:
            return float(self.z_message_gain)
        denominator = max(int(self.message_rounds) - 1 - start_round, 1)
        progress = min(max((int(round_index) - start_round) / float(denominator), 0.0), 1.0)
        return float(self.z_message_gain) + progress * (
            float(self.z_message_gain_final) - float(self.z_message_gain)
        )

    def _edge_z_cavity_signal(self, problem, probabilities, edge_z_message, z_message_gain=None):
        """MaxCut 专用的 Z-consistent directed-edge message。

        直觉：
        - 这里的 z=2p-1 是二值取值 belief，不是物理 Bloch 坐标 Z；
        - MaxCut 要求边两端二值 belief 符号相反；
        - 如果 tail 当前更相信 z_tail=+1，那么 tail->head 应建议 head 更偏向 -1；
        - 为避免一条边两端互相瞬时强化，更新 tail->head 时排除 head->tail 的回流。

        返回：
        - z_error: 节点收到的 Z 建议和当前 Z belief 的差；
        - node_suggestion: 聚合后的 Z 建议；
        - next_message: 下一轮继续使用的有向边消息记忆。
        """
        if problem.edge_index.numel() == 0:
            zeros = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
            return zeros, zeros, edge_z_message

        src, dst = problem.edge_index
        edge_count = int(src.numel())
        tail = torch.cat((src, dst), dim=0)
        head = torch.cat((dst, src), dim=0)
        reverse = torch.cat(
            (
                torch.arange(edge_count, 2 * edge_count, device=self.device),
                torch.arange(0, edge_count, device=self.device),
            ),
            dim=0,
        )
        edge_weight = problem.edge_weight.to(device=self.device, dtype=self.dtype).abs()
        directed_weight = torch.cat((edge_weight, edge_weight), dim=0)
        # z_value 是二值 assignment belief：x=1 -> +1, x=0 -> -1。
        # 它不是 Bloch 向量里的物理 Z 坐标；物理 Z 的读出是 p=(1-Z)/2。
        z_value = 2.0 * probabilities.to(device=self.device, dtype=self.dtype) - 1.0

        if edge_z_message.numel() != 2 * edge_count:
            edge_z_message = -z_value[tail]
        else:
            edge_z_message = edge_z_message.to(device=self.device, dtype=self.dtype)

        incoming = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        incoming.index_add_(0, head, directed_weight * edge_z_message)

        degree = problem.node_degrees(weighted=True, absolute=True).to(device=self.device, dtype=self.dtype)
        cavity_degree = (degree[tail] - directed_weight).clamp_min(1.0)
        cavity_tail_belief = (incoming[tail] - directed_weight * edge_z_message[reverse]) / cavity_degree

        self_mix = torch.as_tensor(
            min(max(float(self.z_message_self_mix), 0.0), 1.0),
            dtype=self.dtype,
            device=self.device,
        )
        if z_message_gain is None:
            z_message_gain = self.z_message_gain
        gain = torch.as_tensor(max(float(z_message_gain), 1e-6), dtype=self.dtype, device=self.device)
        tail_belief = self_mix * z_value[tail] + (1.0 - self_mix) * cavity_tail_belief
        # MaxCut wants opposite Z signs across an edge, so tail->head suggests
        # the negative of tail's non-backtracking cavity belief.
        raw_message = -torch.tanh(gain * tail_belief)

        decay = torch.as_tensor(
            min(max(float(self.z_message_decay), 0.0), 1.0),
            dtype=self.dtype,
            device=self.device,
        )
        next_message = (decay * edge_z_message + (1.0 - decay) * raw_message).clamp(-1.0, 1.0)

        node_suggestion = torch.zeros(problem.num_variables, dtype=self.dtype, device=self.device)
        node_suggestion.index_add_(0, head, directed_weight * next_message)
        node_suggestion = (node_suggestion / degree.clamp_min(1e-6)).clamp(-1.0, 1.0)
        z_error = (node_suggestion - z_value).clamp(-1.0, 1.0)
        return z_error, node_suggestion, next_message

    def _node_step_scale(self, local_field, old_probabilities):
        """可选的节点级步长门控。

        默认返回 1.0，表示所有节点使用同一轮的 field_steps。
        learned_gate 模式会根据局部场强度和当前置信度调节每个节点的 RY 步长。
        """
        if self.node_step_mode != "learned_gate":
            return 1.0
        field_abs = local_field.abs()
        field_norm = field_abs / field_abs.mean().clamp_min(1e-6)
        confidence = 2.0 * (old_probabilities - 0.5).abs()
        logits = (
            self.node_gate_bias.to(device=self.device, dtype=self.dtype)
            + self.node_gate_field.to(device=self.device, dtype=self.dtype) * field_norm
            + self.node_gate_confidence.to(device=self.device, dtype=self.dtype) * confidence
        )
        return (2.0 * torch.sigmoid(logits)).clamp(0.05, 2.5)

    def _xy_feedback_round_scale(self, round_index):
        """Deterministic multiplier for early-on / late-off XY feedback scans."""
        active_fraction = min(max(float(self.xy_feedback_active_fraction), 0.0), 1.0)
        decay_fraction = min(max(float(self.xy_feedback_decay_fraction), 0.0), 1.0)
        active_end = active_fraction * float(self.message_rounds)
        if float(round_index) >= active_end:
            return 0.0
        decay_rounds = decay_fraction * float(self.message_rounds)
        if decay_rounds <= 1e-12:
            return 1.0
        decay_start = max(active_end - decay_rounds, 0.0)
        if float(round_index) < decay_start:
            return 1.0
        return max(0.0, (active_end - float(round_index)) / max(decay_rounds, 1e-12))

    def _final_rotation_angles(self):
        """最后测量前的小全局旋转角，使用 tanh 限幅到 final_rotation_max。"""
        if self.final_rotation_max <= 0.0:
            return torch.zeros(3, dtype=self.dtype, device=self.device)
        limit = torch.as_tensor(self.final_rotation_max, dtype=self.dtype, device=self.device)
        return limit * torch.tanh(self.final_rotation_raw.to(device=self.device, dtype=self.dtype))

    def _apply_final_rotation(self, bloch):
        """在最终读出前对所有节点施加同一个小旋转。"""
        if self.final_rotation_max <= 0.0:
            return bloch
        angles = self._final_rotation_angles().expand(self.num_variables, -1)
        return _apply_bloch_rotation(bloch, angles)

    def _propose_round(
        self,
        problem,
        bloch,
        local_field,
        old_probabilities,
        round_index,
        phase_memory,
        edge_message,
        edge_z_message,
    ):
        """提出第 round_index 轮的 Bloch 更新。

        这一轮不一定会被最终接受；``forward`` 里还会根据 monotone_accept
        检查期望能量是否下降。这里的步骤对应计划书里的 V14-Z-Phase：

        1. 由 local field 和可选 phase memory 形成 RZ 相位信号；
        2. 计算各种邻居/边/cavity 关系信号；
        3. 先做 RZ，相位在 XY 平面积累；
        4. 再做 RY，把局部场和 relation_signal 折回 Z 概率；
        5. 计算 J，必要时 shrink 可疑节点的 proposal。
        """
        # ===== Step 1: RZ 的基础相位信号 =====
        # 默认直接用当前局部场 F_i；memory 模式会把历史局部场累积起来，
        # 让 RZ 相位带一点时间记忆，而不只看当前一轮。
        next_phase_memory = phase_memory
        phase_signal = local_field
        if "memory" in self.phase_mode:
            decay = torch.as_tensor(self.phase_memory_decay, dtype=self.dtype, device=self.device)
            next_phase_memory = decay * phase_memory + local_field
            phase_signal = next_phase_memory

        # ===== Step 2: 计算可选的关系信号 =====
        # 这些信号都是“隐藏相位通道”，不会直接作为最终答案；最终仍然只读 Z 概率。
        neighbor_torque, neighbor_alignment = self._neighbor_xy_signal(problem, bloch)
        cavity_torque, cavity_alignment = self._cavity_xy_signal(problem, bloch)
        edge_cavity_torque, edge_cavity_alignment, next_edge_message = self._edge_cavity_xy_signal(
            problem,
            bloch,
            edge_message,
        )
        # Z-edge 是当前主线：边 i->j 根据 i 的 Z belief 给 j 一个反向 cut 建议。
        z_edge_error, z_edge_suggestion, next_edge_z_message = self._edge_z_cavity_signal(
            problem,
            old_probabilities,
            edge_z_message,
            self._current_z_message_gain(round_index),
        )
        phase_diff_signal = self._phase_difference_signal(problem, bloch)

        # relation_signal 决定 collapse 阶段把哪一种关系信息折回 RY/Z。
        # 注意：phase_mode 字符串只是实验开关；同一份代码支持多条消融路线。
        relation_signal = neighbor_torque
        if "cavity_xy" in self.phase_mode and "edge_cavity_xy" not in self.phase_mode:
            relation_signal = cavity_torque
        if "edge_cavity_xy" in self.phase_mode:
            relation_signal = edge_cavity_torque
        if "z_edge_cavity" in self.phase_mode:
            relation_signal = z_edge_error
        if "phase_diff" in self.phase_mode:
            relation_signal = phase_diff_signal

        # ===== Step 3: 先做 RZ =====
        # RZ 改变 X/Y 平面里的相位，理论上不直接改变 Z 和 p_i。
        phase_angles = torch.zeros_like(bloch)
        phase_angles[:, 0] = self.phase_steps[round_index] * phase_signal
        if "xy_feedback" in self.phase_mode:
            # 把当前 XY 相位 atan2(Y,X) 反馈进下一次 RZ。
            xy_phase = torch.atan2(bloch[:, 1], bloch[:, 0])
            xy_scale = self._xy_feedback_round_scale(round_index)
            phase_angles[:, 0] = phase_angles[:, 0] + xy_scale * self.xy_feedback_steps[round_index] * xy_phase
        if "neighbor_xy" in self.phase_mode:
            # 邻居 XY torque 作为额外 RZ 相位项。
            phase_angles[:, 0] = phase_angles[:, 0] + self.neighbor_phase_steps[round_index] * neighbor_torque
        if "cavity_xy" in self.phase_mode and "edge_cavity_xy" not in self.phase_mode:
            phase_angles[:, 0] = phase_angles[:, 0] + self.neighbor_phase_steps[round_index] * cavity_torque
        if "edge_cavity_xy" in self.phase_mode:
            phase_angles[:, 0] = phase_angles[:, 0] + self.neighbor_phase_steps[round_index] * edge_cavity_torque
        if "phase_diff" in self.phase_mode:
            phase_angles[:, 0] = phase_angles[:, 0] + self.phase_diff_steps[round_index] * phase_diff_signal

        after_rz = _apply_bloch_rotation(bloch, phase_angles)

        # ===== Step 4: 再做 RY / collapse =====
        # RY 是真正改变 Z、从而改变概率 p_i 的操作。
        mixer_angles = torch.zeros_like(bloch)
        step_scale = self._node_step_scale(local_field, old_probabilities)
        # 基础 RY：沿局部场方向更新，field_steps 是每轮可训练步长。
        mixer_angles[:, 1] = self.mixer_bias[round_index] - self.field_steps[round_index] * step_scale * local_field
        if "double_rz" in self.phase_mode:
            # double_rz 消融：再加一个 Z 轴相关角度，作为额外相位通道。
            mixer_angles[:, 2] = self.omega_steps[round_index] * phase_signal
        if "collapse" in self.phase_mode:
            # collapse 通常在后期打开，把 relation_signal 通过 RY 折回 Z 概率。
            start_round = int(round(float(self.message_rounds) * self.two_stage_fraction))
            if round_index >= start_round:
                mixer_angles[:, 1] = mixer_angles[:, 1] + self.collapse_steps[round_index] * relation_signal
        raw_proposal = _apply_bloch_rotation(after_rz, mixer_angles)
        raw_proposal = _apply_bloch_noise(raw_proposal, self.noise_config)

        # ===== Step 5: J / trust region 检查 =====
        # J_i=-F_i*delta_p_i。J_i<0 表示这一步在局部场意义上可能增大能量。
        raw_probabilities = self._probabilities_from_bloch(raw_proposal)
        raw_j = -local_field * (raw_probabilities - old_probabilities)
        proposal = raw_proposal
        trust_active = self.trust_shrink < 1.0 or self.trust_mode in {"adaptive", "two_stage"}
        if self.trust_mode == "two_stage":
            start_round = int(round(float(self.message_rounds) * self.two_stage_fraction))
            trust_active = trust_active and round_index >= start_round
        if trust_active:
            bad = raw_j < -float(self.trust_threshold)
            if self.trust_mode == "adaptive":
                # adaptive 模式：J 越坏，alpha 越小，proposal 越接近旧状态。
                excess = torch.relu(-raw_j - float(self.trust_threshold))
                scale = max(float(self.adaptive_trust_scale), 1e-12)
                alpha = 1.0 / (1.0 + excess / scale)
                alpha = alpha.clamp(min=float(self.adaptive_trust_min), max=1.0)
            else:
                # fixed/two_stage 模式：坏节点统一按 trust_shrink 缩小更新幅度。
                alpha = torch.full_like(raw_j, float(self.trust_shrink))
            shrunk = bloch + alpha.unsqueeze(-1) * (raw_proposal - bloch)
            shrunk = self._safe_project_bloch_ball(shrunk)
            proposal = torch.where(bad.unsqueeze(-1), shrunk, raw_proposal)

        # final_j 记录 trust region 之后的实际更新方向质量。
        proposed_probabilities = self._probabilities_from_bloch(proposal)
        final_j = -local_field * (proposed_probabilities - old_probabilities)
        return proposal, next_phase_memory, next_edge_message, next_edge_z_message, {
            "raw_j": raw_j,
            "j": final_j,
            "after_rz_x": after_rz[:, 0],
            "phase_angle": phase_angles[:, 0],
            "neighbor_torque": neighbor_torque,
            "neighbor_alignment": neighbor_alignment,
            "cavity_torque": cavity_torque,
            "cavity_alignment": cavity_alignment,
            "edge_cavity_torque": edge_cavity_torque,
            "edge_cavity_alignment": edge_cavity_alignment,
            "z_edge_error": z_edge_error,
            "z_edge_suggestion": z_edge_suggestion,
            "phase_diff_signal": phase_diff_signal,
        }

    def forward(self, problem, return_state=False):
        """运行完整 message_rounds 轮 SQNN，并返回最终概率或完整轨迹。

        forward 的核心循环是：

        1. 从当前 Bloch 状态读出概率 p；
        2. 用 p 计算 QUBO local field；
        3. 调用 _propose_round 产生下一步 Bloch proposal；
        4. 如果 monotone_accept=True，只接受期望能量不升的 proposal；
        5. 保存 energy/probability/Bloch/J 等 trace，供训练、画图和排错。
        """
        problem = self._prepare_problem(problem)
        if problem.num_variables != self.num_variables:
            raise ValueError(f"expected {self.num_variables} variables, got {problem.num_variables}")

        # 初态：来自 warm-start 或 |+>，然后可选破对称。
        bloch = self._initial_bloch(problem)
        probabilities = self._probabilities_from_bloch(bloch)
        current_energy = problem.expected_energy(probabilities)

        # trace 用来检查每一轮是否按预期演化。
        energy_trace = [current_energy]
        probability_trace = [probabilities]
        bloch_trace = [bloch]
        accepted_rounds = []
        j_trace = []
        raw_j_trace = []
        after_rz_x_trace = []
        phase_angle_trace = []
        phase_memory = torch.zeros_like(probabilities)
        edge_message = torch.empty(0, dtype=self.dtype, device=self.device)
        edge_z_message = torch.empty(0, dtype=self.dtype, device=self.device)

        for round_index in range(self.message_rounds):
            # 用旧概率计算 local field；本轮所有节点同步更新。
            old_probabilities = probabilities
            local_field = self._local_field(problem, old_probabilities)
            previous_phase_memory = phase_memory
            previous_edge_message = edge_message
            previous_edge_z_message = edge_z_message
            proposed_bloch, phase_memory, edge_message, edge_z_message, diagnostics = self._propose_round(
                problem,
                bloch,
                local_field,
                old_probabilities,
                round_index,
                phase_memory,
                edge_message,
                edge_z_message,
            )
            proposed_probabilities = self._probabilities_from_bloch(proposed_bloch)
            proposed_energy = problem.expected_energy(proposed_probabilities)

            # monotone_accept 是整轮级别的接受/拒绝，不是逐节点接受。
            accepted = True
            if self.monotone_accept:
                accepted = bool((proposed_energy <= current_energy + 1e-9).detach().item())
            if accepted:
                bloch = proposed_bloch
                probabilities = proposed_probabilities
                current_energy = proposed_energy
            elif self.rollback_aux_on_reject:
                phase_memory = previous_phase_memory
                edge_message = previous_edge_message
                edge_z_message = previous_edge_z_message

            energy_trace.append(current_energy)
            probability_trace.append(probabilities)
            bloch_trace.append(bloch)
            accepted_rounds.append(accepted)
            j_trace.append(diagnostics["j"])
            raw_j_trace.append(diagnostics["raw_j"])
            after_rz_x_trace.append(diagnostics["after_rz_x"])
            phase_angle_trace.append(diagnostics["phase_angle"])

        # 最后允许一个小的全局 pre-measurement rotation，再读出最终概率。
        bloch = self._apply_final_rotation(bloch)
        probabilities = self._probabilities_from_bloch(bloch)
        current_energy = problem.expected_energy(probabilities)
        energy_trace[-1] = current_energy
        probability_trace[-1] = probabilities
        bloch_trace[-1] = bloch

        # 防止数值异常污染后续采样/rounding。
        probabilities = torch.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if return_state:
            return {
                "probabilities": probabilities,
                "bloch_state": bloch,
                "expected_energy": problem.expected_energy(probabilities),
                "energy_trace": torch.stack(energy_trace),
                "probability_trace": torch.stack(probability_trace),
                "bloch_trace": torch.stack(bloch_trace),
                "accepted_rounds": accepted_rounds,
                "accepted_mask": torch.tensor(accepted_rounds, device=self.device, dtype=self.dtype),
                "j_trace": torch.stack(j_trace),
                "raw_j_trace": torch.stack(raw_j_trace),
                "after_rz_x_trace": torch.stack(after_rz_x_trace),
                "phase_angle_trace": torch.stack(phase_angle_trace),
                "final_rotation_angles": self._final_rotation_angles(),
            }
        return probabilities


class MultiHeadPhaseAwareSQNN(nn.Module):
    """多头 phase-aware SQNN 实验变体。

    它不是当前最强主线，只是一个消融/候选方向：复制多个
    ``PhaseAwareJRegularizedSQNN`` head，每个 head 用不同 symmetry_seed，
    然后在 logit 空间学习一个加权平均读出。
    """

    def __init__(
        self,
        num_variables,
        message_rounds,
        head_count=3,
        head_seed_stride=7919,
        **head_kwargs,
    ):
        super().__init__()
        self.num_variables = int(num_variables)
        self.message_rounds = int(message_rounds)
        self.head_count = int(head_count)
        self.head_seed_stride = int(head_seed_stride)
        if self.head_count < 1:
            raise ValueError("head_count must be positive")
        base_seed = int(head_kwargs.get("symmetry_seed", 0))
        heads = []
        for index in range(self.head_count):
            item_kwargs = dict(head_kwargs)
            item_kwargs["symmetry_seed"] = base_seed + self.head_seed_stride * index
            heads.append(
                PhaseAwareJRegularizedSQNN(
                    num_variables=self.num_variables,
                    message_rounds=self.message_rounds,
                    **item_kwargs,
                )
            )
        self.heads = nn.ModuleList(heads)
        # head_readout_logits 学习每个 head 在最终概率混合中的权重。
        self.head_readout_logits = nn.Parameter(torch.zeros(self.head_count))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def field_steps(self):
        return torch.stack([head.field_steps for head in self.heads], dim=0).mean(dim=0)

    @property
    def phase_steps(self):
        return torch.stack([head.phase_steps for head in self.heads], dim=0).mean(dim=0)

    @property
    def omega_steps(self):
        return torch.stack([head.omega_steps for head in self.heads], dim=0).mean(dim=0)

    @property
    def xy_feedback_steps(self):
        return torch.stack([head.xy_feedback_steps for head in self.heads], dim=0).mean(dim=0)

    @property
    def neighbor_phase_steps(self):
        return torch.stack([head.neighbor_phase_steps for head in self.heads], dim=0).mean(dim=0)

    @property
    def phase_diff_steps(self):
        return torch.stack([head.phase_diff_steps for head in self.heads], dim=0).mean(dim=0)

    @property
    def collapse_steps(self):
        return torch.stack([head.collapse_steps for head in self.heads], dim=0).mean(dim=0)

    @property
    def mixer_bias(self):
        return torch.stack([head.mixer_bias for head in self.heads], dim=0).mean(dim=0)

    def current_symmetry_strength(self):
        strengths = [head.current_symmetry_strength() for head in self.heads]
        return torch.stack(strengths).mean()

    def _final_rotation_angles(self):
        return torch.stack([head._final_rotation_angles() for head in self.heads], dim=0).mean(dim=0)

    def _prepare_problem(self, problem):
        return self.heads[0]._prepare_problem(problem)

    def _aggregate_probabilities(self, head_probability_trace):
        """在 logit 空间混合多个 head 的 probability trace。

        直接平均概率容易把强置信度 head 稀释；logit 平均会更接近“证据”
        的加权平均。
        """
        weights = torch.softmax(self.head_readout_logits.to(device=self.device, dtype=self.dtype), dim=0)
        clamped = head_probability_trace.clamp(1e-6, 1.0 - 1e-6)
        logits = torch.logit(clamped)
        mixed_logits = (weights.view(-1, 1, 1) * logits).sum(dim=0)
        return torch.sigmoid(mixed_logits)

    def forward(self, problem, return_state=False):
        problem = self._prepare_problem(problem)
        # 先让每个 head 独立跑完整个 SQNN。
        head_states = [head(problem, return_state=True) for head in self.heads]
        head_probability_trace = torch.stack([state["probability_trace"] for state in head_states], dim=0)
        # 再把多个 head 的概率轨迹聚合成一个最终轨迹。
        probability_trace = self._aggregate_probabilities(head_probability_trace)
        probabilities = probability_trace[-1].clamp(0.0, 1.0)
        energy_trace = torch.stack([problem.expected_energy(item) for item in probability_trace])
        bloch_trace = torch.stack([state["bloch_trace"] for state in head_states], dim=0).mean(dim=0)
        raw_j_trace = torch.stack([state["raw_j_trace"] for state in head_states], dim=0).mean(dim=0)
        after_rz_x_trace = torch.stack([state["after_rz_x_trace"] for state in head_states], dim=0).mean(dim=0)
        phase_angle_trace = torch.stack([state["phase_angle_trace"] for state in head_states], dim=0).mean(dim=0)

        j_rows = []
        for round_index in range(1, probability_trace.shape[0]):
            # 聚合后的 J 重新按混合概率计算，方便和单 head 的 J 指标保持一致。
            old_probabilities = probability_trace[round_index - 1]
            new_probabilities = probability_trace[round_index]
            local_field = self.heads[0]._local_field(problem, old_probabilities)
            j_rows.append(-local_field * (new_probabilities - old_probabilities))
        j_trace = torch.stack(j_rows)
        accepted_rounds = [
            all(bool(state["accepted_rounds"][round_index]) for state in head_states)
            for round_index in range(self.message_rounds)
        ]
        if return_state:
            return {
                "probabilities": probabilities,
                "bloch_state": bloch_trace[-1],
                "expected_energy": problem.expected_energy(probabilities),
                "energy_trace": energy_trace,
                "probability_trace": probability_trace,
                "bloch_trace": bloch_trace,
                "accepted_rounds": accepted_rounds,
                "accepted_mask": torch.tensor(accepted_rounds, device=self.device, dtype=self.dtype),
                "j_trace": j_trace,
                "raw_j_trace": raw_j_trace,
                "after_rz_x_trace": after_rz_x_trace,
                "phase_angle_trace": phase_angle_trace,
                "final_rotation_angles": torch.stack(
                    [state.get("final_rotation_angles", torch.zeros(3, device=self.device, dtype=self.dtype)) for state in head_states],
                    dim=0,
                ).mean(dim=0),
            }
        return probabilities


def load_base_config(exploration_dir, run_id):
    model_path = exploration_dir / "runs" / run_id / "model.pt"
    if not model_path.exists():
        return {
            "phase": "maxcut3_phase_base",
            "benchmark": "random_regular_maxcut",
            "n": 512,
            "average_degree": 3.0,
            "seed": 42,
            "noise_rate": 0.10,
            "negative_ratio": 0.50,
            "rounds": 280,
            "epochs": 110,
            "lr": 0.003,
            "weight_decay": 0.0,
            "entropy_weight": 0.02,
            "final_entropy_weight": 0.001,
            "num_samples": 256,
            "local_search_passes": 220,
            "sample_local_search_passes": 80,
            "j_weight": 100.0,
            "penalty": "relu",
            "round_weight": "flat",
            "accepted_only": False,
            "trust_mode": "two_stage",
            "trust_shrink": 0.25,
            "trust_threshold": 1e-4,
            "adaptive_trust_min": 0.0,
            "adaptive_trust_scale": 1e-3,
            "two_stage_fraction": 0.6,
            "symmetry_breaking": "random_ry",
            "symmetry_strength": 0.10,
            "symmetry_strength_trainable": True,
            "symmetry_strength_max": 0.5,
            "symmetry_seed": 42,
            "warm_start_source": "none",
            "warm_start_confidence": 0.0,
            "warm_start_random_samples": 0,
            "warm_start_batch_size": 0,
            "warm_start_local_search_passes": 0,
            "softplus_tau": 1e-3,
            "grad_clip": 1.0,
            "log_every": 10,
        }
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    return dict(payload["config"])


def with_updates(config, **updates):
    item = dict(config)
    item.update(updates)
    return item


def build_variants(base, rounds, epochs):
    common = with_updates(
        base,
        benchmark="random_regular_maxcut",
        n=int(base.get("n", 512)),
        average_degree=float(base.get("average_degree", 3.0)),
        seed=int(base.get("seed", 42)),
        rounds=int(rounds),
        epochs=int(epochs),
        num_samples=256,
        local_search_passes=220,
        sample_local_search_passes=80,
        log_every=10,
        warm_start_source="none",
        phase_mode="baseline",
        phase_memory_decay=0.0,
        xy_feedback_init=0.0,
        omega_init=0.0,
        neighbor_phase_init=0.0,
        phase_diff_init=0.0,
        collapse_init=0.0,
        final_rotation_max=0.0,
        edge_message_decay=0.70,
        edge_message_self_mix=0.50,
        z_message_decay=0.70,
        z_message_self_mix=0.50,
        z_message_gain=1.0,
        z_message_gain_final="",
        z_message_gain_schedule_start=0.60,
        head_count=1,
        head_seed_stride=7919,
        node_step_mode="none",
        vector_loss_weight=0.0,
    )
    variants = [
        (
            "v14_reference_random_ry",
            dict(symmetry_breaking="random_ry"),
        ),
        (
            "v14_memory_xy_reference",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
            ),
        ),
        (
            "v14_neighbor_xy_torque",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="neighbor_xy",
                neighbor_phase_init=0.05,
            ),
        ),
        (
            "v14_memory_neighbor_xy_torque",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_neighbor_xy",
                phase_memory_decay=0.80,
                neighbor_phase_init=0.05,
            ),
        ),
        (
            "v14_neighbor_xy_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="neighbor_xy_collapse",
                neighbor_phase_init=0.05,
                collapse_init=0.03,
            ),
        ),
        (
            "v14_memory_xy_edge_cavity",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_edge_cavity_xy",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                neighbor_phase_init=0.05,
                edge_message_decay=0.70,
                edge_message_self_mix=0.50,
            ),
        ),
        (
            "v14_memory_xy_edge_cavity_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_edge_cavity_xy_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                neighbor_phase_init=0.05,
                collapse_init=0.03,
                edge_message_decay=0.70,
                edge_message_self_mix=0.50,
            ),
        ),
        (
            "v14_memory_xy_z_edge_cavity_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.0,
            ),
        ),
        (
            "v14_memory_xy_neighbor_z_edge_cavity_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_neighbor_xy_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                neighbor_phase_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.0,
            ),
        ),
        (
            "v14_memory_xy_z_edge_decay045_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.45,
                z_message_self_mix=0.50,
                z_message_gain=1.0,
            ),
        ),
        (
            "v14_memory_xy_z_edge_decay085_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.85,
                z_message_self_mix=0.50,
                z_message_gain=1.0,
            ),
        ),
        (
            "v14_memory_xy_z_edge_selfmix025_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.25,
                z_message_gain=1.0,
            ),
        ),
        (
            "v14_memory_xy_z_edge_selfmix075_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.75,
                z_message_gain=1.0,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain06_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=0.6,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain18_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.8,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain14_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.4,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain12_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.2,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain13_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.3,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain15_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.5,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain16_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.6,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain20_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=2.0,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain_schedule_1p0_2p6_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=1.0,
                z_message_gain_final=2.6,
                z_message_gain_schedule_start=0.60,
            ),
        ),
        (
            "v14_memory_xy_z_edge_gain26_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_z_edge_cavity_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                z_message_decay=0.70,
                z_message_self_mix=0.50,
                z_message_gain=2.6,
            ),
        ),
        (
            "v14_multihead_memory_xy",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                head_count=3,
                head_seed_stride=7919,
            ),
        ),
        (
            "v14_multihead_memory_xy_neighbor_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_neighbor_xy_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                neighbor_phase_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                head_count=3,
                head_seed_stride=7919,
            ),
        ),
        (
            "v14_memory_xy_neighbor_collapse_entropy_zero",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_neighbor_xy_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                neighbor_phase_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                entropy_weight=0.01,
                final_entropy_weight=0.0,
            ),
        ),
        (
            "v14_memory_xy_neighbor_collapse_entropy_sharp",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="memory_xy_feedback_neighbor_xy_collapse",
                phase_memory_decay=0.80,
                xy_feedback_init=0.05,
                neighbor_phase_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.05,
                entropy_weight=0.01,
                final_entropy_weight=-0.003,
            ),
        ),
        (
            "v14_phase_diff_torque",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="phase_diff",
                phase_diff_init=0.05,
            ),
        ),
        (
            "v14_phase_diff_collapse",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="phase_diff_collapse",
                phase_diff_init=0.05,
                collapse_init=0.03,
            ),
        ),
        (
            "v14_double_rz_small_final_rotation",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="double_rz",
                omega_init=0.05,
                final_rotation_max=0.08,
            ),
        ),
        (
            "v14_neighbor_xy_small_final_rotation",
            dict(
                symmetry_breaking="random_rz_ry",
                phase_mode="neighbor_xy_collapse",
                neighbor_phase_init=0.05,
                collapse_init=0.03,
                final_rotation_max=0.08,
            ),
        ),
    ]
    return [with_updates(common, phase=name, **updates) for name, updates in variants]


def maxcut_vector_ratio(benchmark, bloch, best_known):
    # Diagnostic only: this scores full Bloch-vector anti-alignment
    # sum w_ij (1 - r_i dot r_j)/2. It is not the measurement-faithful MaxCut
    # objective, whose physical cost is Z-basis C = sum w_ij(1-Z_i Z_j)/2.
    if benchmark.edge_index.numel() == 0:
        return bloch.new_tensor(0.0)
    src, dst = benchmark.edge_index
    weights = benchmark.edge_weight.to(device=bloch.device, dtype=bloch.dtype)
    vectors = F.normalize(bloch, dim=-1, eps=1e-6)
    dot = (vectors[src] * vectors[dst]).sum(dim=-1).clamp(-1.0, 1.0)
    cut_value = (weights * (1.0 - dot) * 0.5).sum()
    known = best_known.to(device=bloch.device, dtype=bloch.dtype).clamp_min(1e-12)
    return cut_value / known


def phase_state_stats(benchmark, state, best_known):
    bloch_trace = state["bloch_trace"]
    ratios = torch.stack([maxcut_vector_ratio(benchmark, item, best_known) for item in bloch_trace[1:]])
    final_bloch = bloch_trace[-1]
    xy_radius = torch.linalg.vector_norm(final_bloch[:, :2], dim=-1).mean()
    final_rotation = state.get("final_rotation_angles")
    if final_rotation is None:
        rotation_norm = 0.0
    else:
        rotation_norm = torch.linalg.vector_norm(final_rotation).detach().cpu()
    return {
        "vector_best_ratio": float(ratios.max().detach().cpu()) if ratios.numel() else 0.0,
        "vector_final_ratio": float(ratios[-1].detach().cpu()) if ratios.numel() else 0.0,
        "final_xy_radius": float(xy_radius.detach().cpu()),
        "final_rotation_norm": float(rotation_norm),
    }


def rewrite_phase_summary(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=PHASE_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PHASE_SUMMARY_FIELDS})


def train_phase_one(config, device, output_dir):
    """训练并评估一个 phase-aware 配置。

    这个函数是单次实验的入口：

    1. 根据 config 构造 MaxCut-3 benchmark；
    2. 构造 warm-start 概率；
    3. 根据 head_count 选择单 head 或 multi-head 模型；
    4. 用 Z-basis expected MaxCut energy + entropy + J penalty 训练；
    5. 调用 evaluate_solution_quality 生成 direct/sample/local-search 指标。

    注意：这里的 ``best_known`` 来自 benchmark.known_optimum。当前
    random_regular_maxcut 里它是总边权 W，所以本函数记录的 ``*_ratio``
    默认是 C/W cut fraction。
    """
    run_id = config_id(config)
    run_dir = output_dir / "runs" / run_id
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        return payload["summary"], True

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(config["seed"]))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config["seed"]) + 1009)

    # benchmark.known_optimum 在当前 MaxCut-3 中是 W=sum_edges w_ij。
    # 因此 best_known 不是严格最优 C*，后续 ratio 字段默认表示 C/W。
    benchmark = make_benchmark(make_train_args(config))
    benchmark.problem = benchmark.problem.to(device=device)
    benchmark.edge_index = benchmark.edge_index.to(device=device)
    benchmark.edge_weight = benchmark.edge_weight.to(device=device, dtype=benchmark.problem.linear.dtype)
    best_known = benchmark.known_optimum.to(device=device, dtype=benchmark.problem.linear.dtype)
    problem = benchmark.problem
    warm_start_probabilities, warm_start_stats = make_warm_start_probabilities(
        config,
        benchmark,
        problem,
        device,
    )

    # 把 config 里的实验开关集中转成模型构造参数。
    model_kwargs = dict(
        trust_mode=config.get("trust_mode", "fixed"),
        trust_shrink=float(config["trust_shrink"]),
        trust_threshold=float(config["trust_threshold"]),
        adaptive_trust_min=float(config.get("adaptive_trust_min", 0.0)),
        adaptive_trust_scale=float(config.get("adaptive_trust_scale", 1e-3)),
        two_stage_fraction=float(config.get("two_stage_fraction", 0.0)),
        symmetry_breaking=config.get("symmetry_breaking", "none"),
        symmetry_strength=float(config.get("symmetry_strength", 0.0)),
        symmetry_strength_trainable=bool(config.get("symmetry_strength_trainable", False)),
        symmetry_strength_max=float(config.get("symmetry_strength_max", 0.5)),
        symmetry_seed=int(config.get("symmetry_seed", config["seed"])),
        initial_probabilities=warm_start_probabilities,
        phase_mode=config.get("phase_mode", "baseline"),
        phase_memory_decay=float(config.get("phase_memory_decay", 0.0)),
        xy_feedback_init=float(config.get("xy_feedback_init", 0.0)),
        xy_feedback_active_fraction=float(config.get("xy_feedback_active_fraction", 1.0)),
        xy_feedback_decay_fraction=float(config.get("xy_feedback_decay_fraction", 0.0)),
        omega_init=float(config.get("omega_init", 0.0)),
        neighbor_phase_init=float(config.get("neighbor_phase_init", 0.0)),
        phase_diff_init=float(config.get("phase_diff_init", 0.0)),
        collapse_init=float(config.get("collapse_init", 0.0)),
        final_rotation_max=float(config.get("final_rotation_max", 0.0)),
        edge_message_decay=float(config.get("edge_message_decay", 0.70)),
        edge_message_self_mix=float(config.get("edge_message_self_mix", 0.50)),
        z_message_decay=float(config.get("z_message_decay", 0.70)),
        z_message_self_mix=float(config.get("z_message_self_mix", 0.50)),
        z_message_gain=float(config.get("z_message_gain", 1.0)),
        z_message_gain_final=(
            None
            if config.get("z_message_gain_final", "") in {"", None}
            else float(config.get("z_message_gain_final"))
        ),
        z_message_gain_schedule_start=float(config.get("z_message_gain_schedule_start", 0.60)),
        node_step_mode=config.get("node_step_mode", "none"),
        rollback_aux_on_reject=bool(config.get("rollback_aux_on_reject", False)),
    )
    if int(config.get("head_count", 1)) > 1:
        # 多头实验路线：多个 symmetry seed 的 head + logit 加权读出。
        model = MultiHeadPhaseAwareSQNN(
            num_variables=problem.num_variables,
            message_rounds=int(config["rounds"]),
            head_count=int(config.get("head_count", 1)),
            head_seed_stride=int(config.get("head_seed_stride", 7919)),
            **model_kwargs,
        ).to(device)
    else:
        # 当前 V14 主线通常走单 head PhaseAwareJRegularizedSQNN。
        model = PhaseAwareJRegularizedSQNN(
            num_variables=problem.num_variables,
            message_rounds=int(config["rounds"]),
            **model_kwargs,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    history = []
    start = time.perf_counter()
    for epoch in range(int(config["epochs"])):
        # ===== 训练一步 =====
        # forward 返回完整 state，是因为 loss 需要最终概率，也需要 J trace。
        optimizer.zero_grad(set_to_none=True)
        state = model(problem, return_state=True)
        probabilities = state["probabilities"]
        energy = problem.expected_energy(probabilities)
        # Main training objective remains Z-basis/product-distribution MaxCut:
        # E_QUBO(p) = -C(p). RZ/XY phase terms are only hidden dynamics unless
        # vector_loss_weight is explicitly set for an auxiliary experiment.
        normalized_energy = energy / (problem.num_variables * problem.coefficient_scale())
        progress = epoch / max(int(config["epochs"]) - 1, 1)
        entropy_weight = float(config["entropy_weight"]) * (1.0 - progress) + float(
            config["final_entropy_weight"]
        ) * progress
        entropy = bernoulli_entropy(probabilities).mean()
        j_penalty = j_penalty_value(state["j_trace"], state["accepted_mask"], config)
        vector_ratio = maxcut_vector_ratio(benchmark, state["bloch_state"], best_known)
        vector_weight = float(config.get("vector_loss_weight", 0.0))
        # Keep vector_weight at 0.0 for the main measurement-faithful route.
        # Nonzero values intentionally add a full-vector auxiliary loss and
        # should be reported separately from the Z-basis mainline.
        loss = (
            (1.0 - vector_weight) * normalized_energy
            - vector_weight * vector_ratio
            - entropy_weight * entropy
            + float(config["j_weight"]) * j_penalty
        )
        loss.backward()
        # 梯度裁剪防止某些角度参数在局部场较强时爆掉。
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        optimizer.step()
        if epoch == 0 or epoch == int(config["epochs"]) - 1 or (epoch + 1) % int(config["log_every"]) == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            # -energy/known is C(p)/known. With known=W it is expected cut
            # fraction; with known=C* it is expected approximation ratio.
            expected_trace_ratio = -state["energy_trace"][1:] / best_known.clamp_min(1e-12)
            history.append(
                {
                    "epoch": int(epoch),
                    "loss": float(loss.detach().cpu()),
                    "normalized_energy": float(normalized_energy.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "entropy_weight": float(entropy_weight),
                    "j_penalty": float(j_penalty.detach().cpu()),
                    "vector_ratio": float(vector_ratio.detach().cpu()),
                    "best_expected_ratio": float(expected_trace_ratio.max().detach().cpu()),
                    "final_expected_ratio": float(expected_trace_ratio[-1].detach().cpu()),
                    "field_step_mean": float(model.field_steps.detach().mean().cpu()),
                    "phase_step_mean": float(model.phase_steps.detach().mean().cpu()),
                    "omega_step_mean": float(model.omega_steps.detach().mean().cpu()),
                    "xy_feedback_mean": float(model.xy_feedback_steps.detach().mean().cpu()),
                    "neighbor_phase_mean": float(model.neighbor_phase_steps.detach().mean().cpu()),
                    "phase_diff_mean": float(model.phase_diff_steps.detach().mean().cpu()),
                    "collapse_mean": float(model.collapse_steps.detach().mean().cpu()),
                    "mixer_bias_mean": float(model.mixer_bias.detach().mean().cpu()),
                    "symmetry_strength": float(model.current_symmetry_strength().detach().cpu()),
                    "final_rotation_norm": float(
                        torch.linalg.vector_norm(model._final_rotation_angles()).detach().cpu()
                    ),
                }
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start
    with torch.no_grad():
        state = model(problem, return_state=True)
    # evaluate_solution_quality 来自 explore_j_regularized_sqnn.py。
    # 它会基于 probability_trace 计算 expected、rounding、sampling、local-search 等结果。
    rows, quality = evaluate_solution_quality(config, state, benchmark, best_known, generator)
    phase_stats = phase_state_stats(benchmark, state, best_known)

    summary = {field: config.get(field) for field in PHASE_SUMMARY_FIELDS if field in config}
    summary.update(
        {
            "run_id": run_id,
            "training_seconds": float(training_seconds),
            "final_symmetry_strength": float(model.current_symmetry_strength().detach().cpu()),
            **warm_start_stats,
            **quality,
            **phase_stats,
        }
    )
    for key in PHASE_SUMMARY_FIELDS:
        summary.setdefault(key, "")

    trace_path = run_dir / "trace_rows.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as file_obj:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "summary": summary,
        },
        run_dir / "model.pt",
    )
    with metrics_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "config": config,
                "summary": summary,
                "history": history,
            },
            file_obj,
            indent=2,
        )
    return summary, False


def write_report(output_dir, rows):
    if not rows:
        return {}
    best_round = max(rows, key=lambda row: float(row.get("best_round_local_search_ratio") or 0.0))
    best_sample = max(rows, key=lambda row: float(row.get("best_sample_local_search_ratio") or 0.0))
    best_expected = max(rows, key=lambda row: float(row.get("best_expected_ratio") or 0.0))
    best_vector = max(rows, key=lambda row: float(row.get("vector_best_ratio") or 0.0))
    sorted_rows = sorted(rows, key=lambda row: float(row.get("best_round_local_search_ratio") or 0.0), reverse=True)
    report = {
        "completed_total": len(rows),
        "best_round_local_search": best_round,
        "best_sample_local_search": best_sample,
        "best_expected": best_expected,
        "best_vector": best_vector,
        "rank_by_round_local_search": [
            {
                "phase": row["phase"],
                "run_id": row["run_id"],
                "best_round_local_search_ratio": row["best_round_local_search_ratio"],
                "best_sample_local_search_ratio": row["best_sample_local_search_ratio"],
                "best_expected_ratio": row["best_expected_ratio"],
                "vector_best_ratio": row.get("vector_best_ratio", ""),
                "final_xy_radius": row.get("final_xy_radius", ""),
            }
            for row in sorted_rows
        ],
    }
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/maxcut3_15h_exploration"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maxcut3_phase_aware_probe"))
    parser.add_argument("--base-run-id", default=BASE_RUN_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--average-degree", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--symmetry-seed", type=int, default=None)
    parser.add_argument("--only-phase", action="append", default=[])
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_config(args.source_dir, args.base_run_id)
    if args.n:
        base["n"] = int(args.n)
    if args.average_degree:
        base["average_degree"] = float(args.average_degree)
    if args.seed is not None:
        base["seed"] = int(args.seed)
        base["symmetry_seed"] = int(args.seed) * 249 + 18
    if args.symmetry_seed is not None:
        base["symmetry_seed"] = int(args.symmetry_seed)
    variants = build_variants(base, args.rounds, args.epochs)
    if args.only_phase:
        wanted = set(args.only_phase)
        variants = [config for config in variants if config["phase"] in wanted]
    summary_path = args.output_dir / "summary.csv"
    summary_rows = load_summary(summary_path) if args.resume else []
    seen = {row["run_id"] for row in summary_rows}

    completed = 0
    for config in variants:
        run_id = config_id(config)
        if run_id in seen:
            continue
        if args.max_runs and completed >= int(args.max_runs):
            break
        print(f"RUN {completed + 1}: {run_id}", flush=True)
        summary, loaded = train_phase_one(config, device, args.output_dir)
        if not loaded:
            summary_rows.append(summary)
            rewrite_phase_summary(summary_path, summary_rows)
            seen.add(summary["run_id"])
        completed += 1

    report = write_report(args.output_dir, summary_rows)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
