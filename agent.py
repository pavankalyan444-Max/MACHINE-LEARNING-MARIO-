import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import MarioCNN
from replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(
        self,
        num_actions=7,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon_start=0.1,
        epsilon_end=0.05,
        epsilon_decay=500000,
        buffer_size=10000,
        batch_size=32
    ):

        self.num_actions = num_actions

        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size

        # =========================
        # DEVICE
        # =========================

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Using device:", self.device)

        if torch.cuda.is_available():
            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )

        # =========================
        # NETWORKS
        # =========================

        self.policy_net = MarioCNN(
            num_actions
        ).to(self.device)

        self.target_net = MarioCNN(
            num_actions
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        # =========================
        # OPTIMIZER
        # =========================

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )

        # =========================
        # REPLAY BUFFER
        # =========================

        self.memory = ReplayBuffer(
            buffer_size
        )

        self.steps_done = 0

    # ============================================================
    # EPSILON
    # ============================================================

    def update_epsilon(self, number_of_actions=1):

        self.steps_done += number_of_actions

        self.epsilon = max(
            self.epsilon_end,

            self.epsilon_start
            - (
                self.steps_done
                / self.epsilon_decay
            )
            * (
                self.epsilon_start
                - self.epsilon_end
            )
        )

    # ============================================================
    # ACTION SELECTION
    # ============================================================

    def act_batch(self, states):

        self.update_epsilon(
            len(states)
        )

        # Convert NumPy → GPU Tensor
        states_tensor = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        # Normalize pixels
        states_tensor /= 255.0

        # =========================
        # DQN FORWARD PASS
        # =========================

        with torch.no_grad():

            q_values = self.policy_net(
                states_tensor
            )

        # GPU → CPU
        q_values_np = q_values.cpu().numpy()

        actions = []

        # =========================
        # EPSILON GREEDY
        # =========================

        for i in range(len(states)):

            if random.random() < self.epsilon:

                # Explore
                action = random.randrange(
                    self.num_actions
                )

            else:

                # Exploit
                action = int(
                    np.argmax(
                        q_values_np[i]
                    )
                )

            actions.append(action)

        return (
            np.asarray(
                actions,
                dtype=np.int64
            ),
            q_values_np
        )

    def act(self, state):

        actions, _ = self.act_batch(
            np.expand_dims(state, axis=0)
        )

        return int(actions[0])

    # ============================================================
    # LEARNING
    # ============================================================

    def learn(self):

        if len(self.memory) < self.batch_size:

            return None

        # =========================
        # SAMPLE REPLAY BUFFER
        # =========================

        batch = self.memory.sample(
            self.batch_size
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        # =========================
        # NUMPY ARRAYS
        # =========================

        states = np.asarray(
            states,
            dtype=np.uint8
        )

        next_states = np.asarray(
            next_states,
            dtype=np.uint8
        )

        # =========================
        # GPU TENSORS
        # =========================

        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        # Normalize
        states /= 255.0
        next_states /= 255.0

        # ========================================================
        # CURRENT Q VALUES
        # ========================================================

        current_q_values = (

            self.policy_net(
                states
            )

            .gather(
                1,
                actions.unsqueeze(1)
            )

            .squeeze(1)
        )

        # ========================================================
        # DOUBLE DQN
        # ========================================================

        with torch.no_grad():

            # Policy network chooses action
            next_policy_q = self.policy_net(
                next_states
            )

            best_next_actions = (
                next_policy_q.argmax(
                    dim=1,
                    keepdim=True
                )
            )

            # Target network evaluates action
            next_target_q = (

                self.target_net(
                    next_states
                )

                .gather(
                    1,
                    best_next_actions
                )

                .squeeze(1)
            )

            target_q_values = (

                rewards

                + self.gamma
                * next_target_q
                * (1 - dones)
            )

        # ========================================================
        # LOSS
        # ========================================================

        loss = F.smooth_l1_loss(
            current_q_values,
            target_q_values
        )

        # ========================================================
        # BACKPROPAGATION
        # ========================================================

        self.optimizer.zero_grad()

        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            10.0
        )

        self.optimizer.step()

        return loss.item()

    # ============================================================
    # TARGET NETWORK
    # ============================================================

    def update_target_network(self):

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )