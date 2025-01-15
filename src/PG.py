import os
import gym
import time
import torch
import numba
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display
from typing import List
from gym.spaces import Discrete, Box
from torch.optim import Adam
from torch.distributions.categorical import Categorical

# ref: https://www.bilibili.com/opus/926231077782028328 https://www.bilibili.com/video/BV1GH4y1V7dF


torch.manual_seed(0)
np.random.seed(0)


def render(env: gym.Env, agent=None) -> None:
    '''
    Overview:
        对agent的表现进行可视化,默认使用随机动作
    Arguments:
        - env:强化学习环境
        - agent:智能体
    '''
    obs = env.reset()
    plt.figure(figsize=(6, 6))
    img = plt.imshow(env.render(mode='rgb_array'))  # only call this once
    for i in range(150):
        img.set_data(env.render(mode='rgb_array'))  # just update the data
        display.display(plt.gcf())
        display.clear_output(wait=True)
        action = env.action_space.sample() if agent is None else agent(obs)
        obs, reward, done, info = env.step(action)
        if done:
            print("失去平衡了, 重新开始吧少年!")
            time.sleep(2.5)
            obs = env.reset()
        if i > 140:
            print("不错哦, 居然成功了!")
    env.close()


def make_env(env_name: str = "CartPole-v0", demo: bool = False):
    '''
    Overview:
        实际建立一个运行环境,返回环境与环境的参数
    Arguments:
        - env_name: 要建立的环境名称
        - demo: 是否进行可视化
    Returns:
        - obs_dim: 观测值的维度
        - ·: 动作的维度
    '''
    env = gym.make(env_name)
    n_acts = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    if demo:
        render(env)
        print(f"action_dim {n_acts} obs_dim {obs_dim}")
    return env, obs_dim, n_acts


# 使用随机动作直观的感知环境
# env, obs_dim, n_acts = make_env(demo=False)
env, obs_dim, n_acts = make_env(demo=True)


def mlp(sizes: List[int], activation=nn.Tanh, output_activation=nn.Identity) -> nn.Sequential:
    '''
    Overview:
        建立一个Multilayer perceptron作为策略函数的核心构成
    Arguments:
        -obs:当前step下的观测值
    Returns:
        一策略分布，本质上是一个分布函数，softmax(logits)
    '''
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


hidden_sizes = [32]

logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])


def policy(obs: torch.tensor) -> torch.distributions:
    '''
    Overview:
        策略函数，对应了上文的函数\pi,输入一个obs,返回一个分布，对应了第五节中的公式
    Arguments:
        -obs:当前step下的观测值
    Returns:
        -策略分布，本质上是一个分布函数，softmax(logits)
    '''
    logits = logits_net(obs)
    return Categorical(logits=logits)


def compute_loss(obs: torch.tensor, act: torch.tensor, returns: torch.tensor) -> torch.tensor:
    '''
    Overview:
        计算损失函数,对应了上文的函数 L
    Arguments:
        - obs: 当前step下的观测值
        - act: 当前step下的动作值
        - returns: 当前动作对应的“好坏”评价
    Returns:
        - L: 策略对应的Loss值
    '''
    # 当前的策略给出的动作分布
    policy_distribution = policy(obs)
    # 实际选择的动作 a_t 在当前策略下的logp
    log_probability = policy_distribution.log_prob(act)
    # 按回报的大小强化相应的动作
    L = -(log_probability * returns).mean()
    return L


def train(env_name: str = 'CartPole-v0', lr: float = 1e-2, epochs: int = 50, batch_size: int = 4000, verbose: int = 0):
    '''
    Overview:
        建立一个新的强化学习环境,并在其上训练一个Agent最大化累积reward,使用policy gradient算法,采用value-function作为baseline
        注意三个层次的描述: epoches[episodes[games[(s, a, r, s', d)...]]]

    Arguments:
        - env_name: 环境名称,在其上训练agent
        - lr: 学习率
        - epochs: 训练次数 total step = batch_size * epochs
        - hidden_sizes: 网络隐藏层大小
        - verbose: 0 会展示进度条 1 打印所有信息(只建议debug时使用)

    Returns:
        - logits_net torch.nn.Module: 训练好的策略网略
        - result: 训练过程中的监控指标 List[Tuple(epoch: int 轮次, loss: float 策略损失, return: float 每句游戏平均回报, ep_len: float 每局游戏平均长度)]
    '''

    # 初始化环境
    env, obs_dim, n_acts = make_env(env_name)

    # 定义优化器,将策略网络的参数传入, \theta
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # 从分布中采样
    def get_action(obs):
        return policy(obs).sample().item()

    # 训练一个轮次
    def train_one_epoch():
        # 初始化空列表储存信息
        batch_obs = []  # 存储观测值序列
        batch_acts = []  # 存储动作序列
        batch_weights = []  # 存储 R(\tau) 作为“好坏的评价”用于更新策略
        batch_rets = []  # 存储每局游戏的回报
        batch_lens = []  # 存储每局游戏的长度

        # 训练开始前记得重置参数
        obs = env.reset()
        done = False
        ep_rews = []  # 每局游戏中每步的奖励,注意与回报的区别 returns = sum(reward)

        # 使用当前策略与环境交互,存储观测值,奖励等信息
        while True:

            # 将观测值加入buffer
            batch_obs.append(obs.copy())

            # 使用策略函数给出当前观测值对应的动作,在环境中步进
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))

            obs, rew, done, _ = env.step(act)

            # 将动作和奖励加入buffer
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # episode结束后记录相关参数
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # 更新时某个动作的权重 动作好坏的评价 -> R(tau), \hat{R}(tau), R - V, Q - V ...
                batch_weights += [ep_ret] * ep_len

                # 每个episode后重置参数
                obs, done, ep_rews = env.reset(), False, []

                # 收集满 batch_size 个数据后结束本轮次
                if len(batch_obs) > batch_size:
                    break

        # 反向传播求梯度,进行一次参数的更新
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  returns=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # 训主练循环
    result = []
    for i in tqdm(range(epochs)):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        if verbose:
            print(f'epoch: {i} loss: {batch_loss} return: {np.mean(batch_rets)} ep_len: {np.mean(batch_lens)}')
        result.append((i, batch_loss.item(), np.mean(batch_rets), np.mean(batch_lens)))
    return logits_net, result


logits_net, result = train()

plt.plot(np.array(result)[:, -1])
plt.title("Returns")
plt.legend()
plt.grid()


def agent(obs, logits_net):
    '''
    agent 观测到一个obs 返回一个动作 action
    注意: policy返回的是分布, 而agent返回的是动作
    '''
    # 实际测试时使用策略输出的最大值即可
    action = logits_net(torch.as_tensor(obs, dtype=torch.float32)).argmax().item()
    return action


render(env, lambda obs: agent(obs, logits_net))


@numba.njit
def reward_to_go_jit(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in np.arange(n - 1)[::-1]:
        rtgs[i] = rews[i] + rtgs[i + 1]
    return rtgs


def trainNewModel(env_name: str = 'CartPole-v0', lr: float = 1e-2, epochs: int = 50, batch_size: int = 4000, hidden_sizes: List[int] = [32, ], verbose: int = 0):
    '''
    Overview:
        建立一个新的强化学习环境,并在其上训练一个Agent最大化累积reward,使用policy gradient算法,采用value-function作为baseline

    Arguments:
        - env_name: 环境名称,在其上训练agent
        - lr: 学习率
        - epochs: 训练次数 total step = batch_size * epochs
        - hidden_sizes: 网络隐藏层大小
        - verbose: 0 会展示进度条 1 打印所有信息(只建议debug时使用)

    Returns:
        - logits_net torch.nn.Module: 训练好的策略网略
        - result: 训练过程中的监控指标 List[Tuple(epoch: int 轮次, loss: float 策略损失, return: float 每句游戏平均回报, ep_len: float 每局游戏平均长度)]
    '''
    # 初始化环境
    env, obs_dim, n_acts = make_env(env_name)

    # 构建策略网络
    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
    # 构建价值网络
    value_net = mlp(sizes=[obs_dim] + hidden_sizes + [1])

    # 定义策略, 注意 policy 是一个分布
    def policy(obs: torch.tensor) -> torch.distributions:
        '''
        输入一个obs, 返回一个分布
        '''
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # 定义损失函数
    def compute_loss(obs: torch.tensor, act: torch.tensor, returns: torch.tensor) -> torch.tensor:
        '''
        计算损失函数
        '''
        # 当前的策略给出的动作分布
        policy_distribution = policy(obs)
        # 实际选择的动作 a_t 在当前策略下的logp
        log_probability = policy_distribution.log_prob(act)
        # 按回报的大小强化相应的动作
        L = -(log_probability * returns).mean()
        return L

    def v_loss(obs: torch.tensor, returns: torch.tensor) -> torch.tensor:
        v_pred = value_net(obs)  # 价值网络对当前状态value的预测
        loss = (0.5 * (v_pred - returns) ** 2).mean()  # 计算值函数估计的loss
        return loss

    # 定义优化器,将策略网络的参数传入, \theta
    optimizer_pi = Adam(logits_net.parameters(), lr=lr)
    optimizer_v = Adam(value_net.parameters(), lr=lr)

    # 从分布中采样
    def get_action(obs):
        return policy(obs).sample().item()

    # 训练一个轮次
    def train_one_epoch():
        '''
        Overview:
            利用当前的策略,在环境中进行交互(可能进行了多轮游戏),收集batch_size个数据,在其上更新策略与值函数

        Returns:
            - batch_loss: 当前轮次下策略的Loss值
            - batch_rets: 当前轮次下的每局游戏中策略得到的奖励序列
            - batch_lens: 当前轮次下每局游戏的长度

        '''
        # 初始化空列表储存信息
        batch_obs = []  # 存储观测值序列
        batch_acts = []  # 存储动作序列
        batch_weights = []  # 存储 R(\tau) 作为“好坏的评价”用于更新策略
        batch_rets = []  # 存储每局游戏的回报
        batch_lens = []  # 存储每局游戏的长度

        # 训练开始前记得重置参数
        obs = env.reset()
        done = False
        ep_rews = []  # 每局游戏中每步的奖励,注意与回报的区别 returns = sum(reward)

        # 使用当前策略与环境交互,存储观测值,奖励等信息
        while True:

            # 将观测值加入buffer
            batch_obs.append(obs.copy())

            # 使用策略函数给出当前观测值对应的动作,在环境中步进
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))

            obs, rew, done, _ = env.step(act)

            # 将动作和奖励加入buffer
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # episode结束后记录相关参数
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # 更新时某个动作的权重 动作好坏的评价 -> R(tau), \hat{R}(tau), R - V, Q - V GAE ...
                batch_weights += list(reward_to_go_jit(np.array(ep_rews)))

                # 每个episode后重置参数
                obs, done, ep_rews = env.reset(), False, []

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # 反向传播求梯度,进行一次参数的更新
        # 首先更新policy
        optimizer_pi.zero_grad()
        obs = torch.as_tensor(batch_obs, dtype=torch.float32)
        R = torch.as_tensor(batch_weights, dtype=torch.float32)
        _R = (R - value_net(obs).detach())
        batch_loss = compute_loss(obs=obs,
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  returns=_R
                                  )
        batch_loss.backward()
        optimizer_pi.step()

        # 对value-function进行更新
        optimizer_v.zero_grad()

        value_loss = v_loss(obs=obs, returns=R)
        value_loss.backward()
        optimizer_v.step()

        return batch_loss, batch_rets, batch_lens

    # 主训练循环
    result = []
    for i in tqdm(range(epochs)):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        if verbose:
            print(f'epoch: {i} loss: {batch_loss} return: {np.mean(batch_rets)} ep_len: {np.mean(batch_lens)}')
        result.append((i, batch_loss.item(), np.mean(batch_rets), np.mean(batch_lens)))

    return logits_net, result


logits_net2, result2 = trainNewModel()

plt.plot(np.array(result2)[:, -1], label="improve")
plt.plot(np.array(result)[:, -1], label="origin")
plt.title("Returns")
plt.legend()
plt.grid()

plt.plot(np.array(result2)[:, 1], label="improve")
plt.plot(np.array(result)[:, 1], label="origin")
plt.title("Loss")
plt.legend()
plt.grid()
