# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .custom.humanoid_config import XBotLCfg, XBotLCfgPPO
from .custom.humanoid_arms_config import XBotLArmsCfg, XBotLArmsCfgPPO

from .custom.humanoid_env import XBotLFreeEnv
from .custom.humanoid_arms_env import XBotLArmsFreeEnv

# H1
from .custom.h1_config import H1Cfg, H1CfgPPO
from .custom.h1_env import H1FreeEnv

# H1Dance
from .custom.h1_dance_config import H1DanceCfg, H1DanceCfgPPO
from .custom.h1_dance_env import H1DanceFreeEnv

from humanoid.utils.task_registry import task_registry


task_registry.register( "humanoid_ppo", XBotLFreeEnv, XBotLCfg(), XBotLCfgPPO() )
task_registry.register( "humanoid_arms_ppo", XBotLArmsFreeEnv, XBotLArmsCfg(), XBotLArmsCfgPPO() )
task_registry.register( "h1_ppo", H1FreeEnv, H1Cfg(), H1CfgPPO() )
task_registry.register( "h1_dance", H1DanceFreeEnv, H1DanceCfg(), H1DanceCfgPPO() )
