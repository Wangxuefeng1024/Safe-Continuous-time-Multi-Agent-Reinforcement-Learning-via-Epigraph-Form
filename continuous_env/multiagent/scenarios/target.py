import numpy as np
from continuous_env.multiagent.core import World, Agent, Landmark
from continuous_env.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, seed):
        world = World(seed)
        world.dim_c = 2
        num_agents = 2
        world.collaborative = True
        self.num_agents = num_agents
        # Add agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent {i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.02

        # Add 2 endpoints for line + obstacles
        num_goals = 2
        num_obs = 1
        world.landmarks = [Landmark() for _ in range(num_goals + num_obs)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'landmark {i}'
            landmark.collide = False
            landmark.movable = False
            if i < num_agents:
                landmark.color = np.array([0.25, 0.95, 0.25])
                landmark.size = 0.02
            else:
                landmark.color = np.array([0.8, 0.2, 0.2])
                landmark.size = 0.05

        self.reset_world(world)
        self.goals = self.landmark2goal(world)  # Precompute goals for agents
        return world

    def reset_world(self, world):
        area_size = 1.5

        # --- agent 配色 ---
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])

        # --- landmark 配色（前2个是端点，后面是障碍物） ---
        for i, landmark in enumerate(world.landmarks):
            if i < self.num_agents:
                landmark.color = np.array([0.25, 0.95, 0.25])
                landmark.size = 0.02
            else:
                landmark.color = np.array([0.8, 0.2, 0.2])
                landmark.size = 0.05

        # ========= 固定布局 =========
        # 1) 端点（两端形成长度≈0.5的线段）
        #    p1 = p0 + (0.5/√2, 0.5/√2)  => 线段长度约 0.5
        p0 = np.array([0.00, 0.80], dtype=np.float32)
        p1 = np.array([0.80, 0.80], dtype=np.float32)

        world.landmarks[0].state.p_pos = p0.copy()
        world.landmarks[1].state.p_pos = p1.copy()
        world.landmarks[0].state.p_vel = np.zeros(2, dtype=np.float32)
        world.landmarks[1].state.p_vel = np.zeros(2, dtype=np.float32) 

        # 2) 障碍物（围着线段布置，但与端点/智能体保持 ≥0.15 的间隔）
        obs_positions = [
            # np.array([-1.00, 0.00], dtype=np.float32),  # 在线段附近
            np.array([0.50, 0.50], dtype=np.float32)
        ]
        for i, pos in enumerate(obs_positions, start=self.num_agents):
            world.landmarks[i].state.p_pos = pos.copy()
            world.landmarks[i].state.p_vel = np.zeros(2, dtype=np.float32)

        # 3) 智能体初始位置（左下区域，互不重叠，且与端点/障碍物都有 ≥0.15 距离）
        agent_positions = [
            np.array([0.50, -0.50], dtype=np.float32),
            np.array([0.00, -0.50], dtype=np.float32),
        ]
        for agent, pos in zip(world.agents, agent_positions):
            agent.state.p_pos = pos.copy()
            agent.state.p_vel = np.zeros(2, dtype=np.float32)
            agent.state.c = np.zeros(world.dim_c, dtype=np.float32)

    def reward(self, agent, world):
        idx = world.agents.index(agent)
        # goals = self.landmark2goal(world)
        # goal = self.goals[idx % len(self.goals)]
        goal = self.goals

        dist = np.linalg.norm(agent.state.p_pos - goal[idx])
        rew = -dist

        dis_rew = 0
        # for other in world.agents:
        #     if other is agent:
        #         continue
        #     if self.is_collision(agent, other):
        #         dis_rew -= 15
        for obs in world.landmarks[self.num_agents:]:
            if self.is_collision(agent, obs):
                dis_rew = -10
        continuous_rew = self.continuous_penalty(world, agent)

        return rew, dis_rew, continuous_rew


    def continuous_penalty(self, world, agent, collision_scale=20.0, eps=1e-9):
        """
        Continuous obstacle-only penalty.

        For every agent–obstacle pair:
        penetration = (r_agent + r_obs) - distance(agent, obstacle)
        if penetration > 0 (i.e., collision/overlap):
            add collision_scale * penetration
        else:
            add 0

        No negative contributions when no collision; no agent–agent terms.
        """
        idx = world.agents.index(agent)

        total = 0.0
        obstacles = world.landmarks[self.num_agents:]  # obstacles only
        # for agent in world.agents:
        pa = world.agents[idx].state.p_pos
        ra = world.agents[idx].size
        for obs in obstacles:
            po = obs.state.p_pos
            ro = obs.size
            d = float(np.linalg.norm(pa - po) + eps)
            penetration = (ra + ro) - d
            if penetration > 0.0:
                penetration = collision_scale * penetration
            elif penetration <= 0.0:
                penetration = 2 * penetration
            total += penetration
        return total/ 2.0  # Avoid double counting (i,j) and (j,i)

    # def continuous_penalty(self, world, collision_scale=20.0):
    #     """
    #     连续版 penalty：所有 agent 两两距离的和；若发生碰撞（距离 < 半径和），
    #     则把该对距离乘以一个较大的系数 collision_scale 以夸张惩罚。
    #     注意：按你的要求用双重循环累加，最后除以 2，避免双计数。
    #     """
    #     agents = world.agents
    #     n = len(agents)
    #     total = 0.0
    #     eps = 1e-9
    #     boundary = agents[0].size * 2.0  # 假设所有 agent 半径相同
    #     for i in range(n):
    #         pi = agents[i].state.p_pos
    #         ri = agents[i].size

    #         for j in range(n):
    #             if j == i:
    #                 continue
    #             pj = agents[j].state.p_pos
    #             rj = agents[j].size

    #             d = float(np.linalg.norm(pi - pj) + eps)  # 距离
    #             term = boundary - d
    #             # 若“碰撞”（小于半径和），夸张惩罚
    #             if d < (ri + rj):
    #                 term *= collision_scale
    #             total += term

    #     return total / 2.0  # 因为(i,j)和(j,i)都算了

    def landmark2goal(self, world):
        p0 = world.landmarks[0].state.p_pos
        p1 = world.landmarks[1].state.p_pos
        direction = p1 - p0
        goals = [p0 + direction * (i + 1) / (len(world.agents) + 1) for i in range(len(world.agents))]
        p0 = np.array([0.00, 0.80], dtype=np.float32)
        p1 = np.array([0.80, 0.80], dtype=np.float32)
        return p0, p1

    def is_collision(self, entity1, entity2):
        delta_pos = entity1.state.p_pos - entity2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = entity1.size + entity2.size
        return dist < dist_min

    def observation(self, agent, world):
        idx = world.agents.index(agent)

        rel_landmarks = [
            self.goals[idx] - agent.state.p_pos                  # num_goals == 2
        ]

        rel_obstacles = [
            lm.state.p_pos - agent.state.p_pos
            for lm in world.landmarks[self.num_agents:]          # num_obs == 3
        ]

        other_pos = [a.state.p_pos - agent.state.p_pos for a in world.agents if a is not agent]

        # one-hot for agent identity
        one_hot_id = np.zeros(self.num_agents, dtype=np.float32)
        one_hot_id[idx] = 1.0

        return np.concatenate(
            [agent.state.p_vel, agent.state.p_pos]
            + rel_landmarks
            + rel_obstacles
            + other_pos
            + [one_hot_id]   # 注意要放在 list 里
        )