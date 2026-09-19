"""量化器 init_state 状态机常量（各后端共享，避免魔法数字）。"""

# 未初始化：首批前向将按真实数据初始化 scale / beta
INIT_STATE_UNINIT = 0
# 已初始化：训练阶段按 EMA / 梯度更新量化参数
INIT_STATE_TRAINING = 1
# 已冻结：校准完成或加载 checkpoint 后，不再更新量化参数
INIT_STATE_FROZEN = 999999
