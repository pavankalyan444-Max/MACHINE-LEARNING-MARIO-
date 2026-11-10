import cv2
import numpy as np
import math
import time


class BrainVisualizer:

    # =========================================================
    # ACTION NAMES
    # =========================================================

    ACTION_NAMES = [
        "NOOP",
        "RIGHT",
        "RIGHT + A",
        "RIGHT + B",
        "RIGHT + A + B",
        "A",
        "LEFT"
    ]

    # =========================================================
    # COLORS
    # =========================================================

    BLACK = (4, 6, 10)
    WHITE = (245, 250, 255)

    CYAN = (255, 220, 70)
    BLUE = (255, 150, 30)

    GREEN = (80, 255, 120)

    YELLOW = (80, 240, 255)

    RED = (80, 80, 255)

    GREY = (80, 95, 110)

    # =========================================================
    # INIT
    # =========================================================

    def __init__(
        self,
        num_actions=7,
        num_envs=10,
        width=1600,
        height=900
    ):

        self.num_actions = num_actions
        self.num_envs = num_envs
        self.running = True

        # -----------------------------------------------------
        # Window
        # -----------------------------------------------------

        self.width = width
        self.height = height

        self.window_name = (
            "SUPER MARIO AI - NEURAL BRAIN"
        )

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            self.window_name,
            self.width,
            self.height
        )
        self.window_open = True

        # -----------------------------------------------------
        # Animation
        # -----------------------------------------------------

        self.animation = 0.0

        self.last_time = time.time()

        # -----------------------------------------------------
        # Fake neural activity
        # -----------------------------------------------------

        self.fake_activity = np.random.uniform(
            0.25,
            1.0,
            100
        ).astype(np.float32)

        # -----------------------------------------------------
        # Particle phases
        # -----------------------------------------------------

        self.particle_phases = np.random.uniform(
            0.0,
            1.0,
            200
        ).astype(np.float32)

    # =========================================================
    # TEXT
    # =========================================================

    def draw_text(
        self,
        canvas,
        text,
        x,
        y,
        size=0.5,
        thickness=1,
        color=WHITE
    ):

        cv2.putText(
            canvas,
            str(text),
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            color,
            thickness,
            cv2.LINE_AA
        )

    # =========================================================
    # FAKE NODE
    # =========================================================

    def fake_strength(self, index):

        # Smooth fake neural activity
        wave = math.sin(
            self.animation * 8.0
            + index * 0.73
        )

        noise = np.random.uniform(
            -0.08,
            0.08
        )

        value = (
            0.55
            + wave * 0.25
            + noise
        )

        return float(
            np.clip(
                value,
                0.08,
                1.0
            )
        )

    # =========================================================
    # GLOW CIRCLE
    # =========================================================

    def glow_circle(
        self,
        canvas,
        x,
        y,
        radius,
        intensity,
        selected=False
    ):

        intensity = float(
            np.clip(
                intensity,
                0.0,
                1.0
            )
        )

        # -----------------------------------------------------
        # Glow
        # -----------------------------------------------------

        glow_layer = np.zeros_like(
            canvas
        )

        glow_radius = int(
            radius * (
                2.5
                + intensity * 3.5
            )
        )

        cv2.circle(
            glow_layer,
            (int(x), int(y)),
            glow_radius,
            self.CYAN,
            -1
        )

        alpha = (
            0.025
            + intensity * 0.055
        )

        cv2.addWeighted(
            glow_layer,
            alpha,
            canvas,
            1.0 - alpha,
            0,
            canvas
        )

        # -----------------------------------------------------
        # Main node
        # -----------------------------------------------------

        if selected:

            node_color = self.GREEN

        else:

            brightness = int(
                100
                + intensity * 155
            )

            node_color = (
                brightness,
                brightness,
                255
            )

        cv2.circle(
            canvas,
            (int(x), int(y)),
            radius,
            node_color,
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            canvas,
            (int(x), int(y)),
            radius + 1,
            self.CYAN,
            1,
            cv2.LINE_AA
        )

    # =========================================================
    # SIGNAL PARTICLE
    # =========================================================

    def draw_signal(
        self,
        canvas,
        p1,
        p2,
        strength,
        phase
    ):

        strength = float(
            np.clip(
                strength,
                0.0,
                1.0
            )
        )

        # -----------------------------------------------------
        # Connection
        # -----------------------------------------------------

        thickness = (
            1
            if strength < 0.55
            else 2
        )

        connection_color = (
            255,
            int(
                70
                + strength * 150
            ),
            int(
                30
                + strength * 80
            )
        )

        cv2.line(
            canvas,
            p1,
            p2,
            connection_color,
            thickness,
            cv2.LINE_AA
        )

        # -----------------------------------------------------
        # Moving particle
        # -----------------------------------------------------

        position = (
            phase
            + strength * 0.15
        ) % 1.0

        px = int(
            p1[0]
            + (
                p2[0] - p1[0]
            ) * position
        )

        py = int(
            p1[1]
            + (
                p2[1] - p1[1]
            ) * position
        )

        radius = int(
            2
            + strength * 3
        )

        cv2.circle(
            canvas,
            (px, py),
            radius + 3,
            self.CYAN,
            -1
        )

        cv2.circle(
            canvas,
            (px, py),
            radius,
            self.WHITE,
            -1
        )

    # =========================================================
    # DRAW LAYER
    # =========================================================

    def draw_layer(
        self,
        canvas,
        name,
        x,
        center_y,
        node_count=10,
        selected_index=None,
        seed=0
    ):

        spacing = 28

        total_height = (
            node_count - 1
        ) * spacing

        start_y = (
            center_y
            - total_height / 2
        )

        positions = []

        for i in range(node_count):

            px = int(x)

            py = int(
                start_y
                + i * spacing
            )

            positions.append(
                (px, py)
            )

            # Fake activity
            strength = self.fake_strength(
                seed + i
            )

            selected = (
                selected_index == i
            )

            self.glow_circle(
                canvas,
                px,
                py,
                6,
                strength,
                selected
            )

        # -----------------------------------------------------
        # Layer title
        # -----------------------------------------------------

        self.draw_text(
            canvas,
            name,
            x - 35,
            int(start_y - 18),
            0.43,
            1,
            self.WHITE
        )

        return positions

    # =========================================================
    # INPUT LAYER
    # =========================================================

    def draw_input_layer(
        self,
        canvas,
        x,
        center_y
    ):

        node_count = 12

        spacing = 28

        total_height = (
            node_count - 1
        ) * spacing

        start_y = (
            center_y
            - total_height / 2
        )

        positions = []

        for i in range(node_count):

            px = int(x)

            py = int(
                start_y
                + i * spacing
            )

            positions.append(
                (px, py)
            )

            strength = self.fake_strength(
                i
            )

            self.glow_circle(
                canvas,
                px,
                py,
                6,
                strength
            )

        self.draw_text(
            canvas,
            "INPUT",
            x - 22,
            int(start_y - 18),
            0.43,
            1,
            self.WHITE
        )

        return positions

    # =========================================================
    # DENSE LAYER
    # =========================================================

    def draw_dense_layer(
        self,
        canvas,
        x,
        center_y
    ):

        node_count = 14

        spacing = 26

        total_height = (
            node_count - 1
        ) * spacing

        start_y = (
            center_y
            - total_height / 2
        )

        positions = []

        for i in range(node_count):

            px = int(x)

            py = int(
                start_y
                + i * spacing
            )

            positions.append(
                (px, py)
            )

            strength = self.fake_strength(
                40 + i
            )

            self.glow_circle(
                canvas,
                px,
                py,
                6,
                strength
            )

        self.draw_text(
            canvas,
            "DENSE",
            x - 24,
            int(start_y - 18),
            0.43,
            1,
            self.WHITE
        )

        return positions

    # =========================================================
    # CONNECTIONS
    # =========================================================

    def connect_layers(
        self,
        canvas,
        layer_a,
        layer_b,
        offset
    ):

        for i, p1 in enumerate(layer_a):

            # Only 2 connections per node
            for j in range(2):

                target_index = (
                    i * len(layer_b)
                    // len(layer_a)
                    + j
                ) % len(layer_b)

                p2 = layer_b[
                    target_index
                ]

                strength = self.fake_strength(
                    i * 3 + j
                )

                phase = (
                    self.animation
                    + offset
                    + i * 0.07
                    + j * 0.11
                ) % 1.0

                self.draw_signal(
                    canvas,
                    p1,
                    p2,
                    strength,
                    phase
                )

    # =========================================================
    # Q OUTPUT
    # =========================================================

    def draw_q_output(
        self,
        canvas,
        q_values,
        actions,
        x,
        y
    ):

        width = 500
        height = 310

        # -----------------------------------------------------
        # Panel
        # -----------------------------------------------------

        cv2.rectangle(
            canvas,
            (x, y),
            (
                x + width,
                y + height
            ),
            (70, 100, 125),
            1,
            cv2.LINE_AA
        )

        self.draw_text(
            canvas,
            "Q-VALUE DECISION",
            x + 20,
            y + 32,
            0.65,
            2,
            self.WHITE
        )

        self.draw_text(
            canvas,
            "POLICY NETWORK OUTPUT",
            x + 20,
            y + 53,
            0.36,
            1,
            (160, 180, 195)
        )

        if q_values is None:
            return

        q = np.asarray(
            q_values,
            dtype=np.float32
        )

        # -----------------------------------------------------
        # Average environments
        # -----------------------------------------------------

        if q.ndim == 2:

            avg_q = np.mean(
                q,
                axis=0
            )

        else:

            avg_q = q

        best_action = int(
            np.argmax(avg_q)
        )

        q_min = float(
            np.min(avg_q)
        )

        q_max = float(
            np.max(avg_q)
        )

        q_range = max(
            q_max - q_min,
            1e-6
        )

        # -----------------------------------------------------
        # Rows
        # -----------------------------------------------------

        for i in range(
            min(
                self.num_actions,
                len(avg_q)
            )
        ):

            value = float(
                avg_q[i]
            )

            row_y = (
                y
                + 82
                + i * 31
            )

            normalized = (
                value - q_min
            ) / q_range

            bar_width = int(
                normalized * 260
            )

            selected = (
                i == best_action
            )

            # -------------------------------------------------
            # Selected background
            # -------------------------------------------------

            if selected:

                cv2.rectangle(
                    canvas,
                    (
                        x + 8,
                        row_y - 20
                    ),
                    (
                        x + width - 8,
                        row_y + 7
                    ),
                    (20, 45, 32),
                    -1
                )

                cv2.rectangle(
                    canvas,
                    (
                        x + 8,
                        row_y - 20
                    ),
                    (
                        x + width - 8,
                        row_y + 7
                    ),
                    self.GREEN,
                    1
                )

            # -------------------------------------------------
            # Action
            # -------------------------------------------------

            self.draw_text(
                canvas,
                self.ACTION_NAMES[i],
                x + 20,
                row_y,
                0.40,
                1,
                self.GREEN
                if selected
                else self.WHITE
            )

            # -------------------------------------------------
            # Bar
            # -------------------------------------------------

            if bar_width > 0:

                cv2.rectangle(
                    canvas,
                    (
                        x + 155,
                        row_y - 12
                    ),
                    (
                        x + 155 + bar_width,
                        row_y + 1
                    ),
                    self.GREEN
                    if selected
                    else self.CYAN,
                    -1
                )

            # -------------------------------------------------
            # Value
            # -------------------------------------------------

            self.draw_text(
                canvas,
                f"{value:7.2f}",
                x + 430,
                row_y,
                0.40,
                1,
                self.GREEN
                if selected
                else self.WHITE
            )

        # -----------------------------------------------------
        # AI decision
        # -----------------------------------------------------

        self.draw_text(
            canvas,
            "AI DECISION:",
            x + 20,
            y + height - 25,
            0.40,
            1,
            (170, 190, 205)
        )

        self.draw_text(
            canvas,
            self.ACTION_NAMES[
                best_action
            ],
            x + 130,
            y + height - 25,
            0.52,
            2,
            self.GREEN
        )

    # =========================================================
    # MARIO PANEL
    # =========================================================

    def draw_mario_panel(
        self,
        canvas,
        frame,
        x,
        y,
        width,
        height,
        index,
        action,
        reward
    ):

        # -----------------------------------------------------
        # Border
        # -----------------------------------------------------

        cv2.rectangle(
            canvas,
            (x, y),
            (
                x + width,
                y + height
            ),
            (65, 80, 95),
            1
        )

        # -----------------------------------------------------
        # Header
        # -----------------------------------------------------

        self.draw_text(
            canvas,
            f"MARIO {index + 1}",
            x + 8,
            y + 20,
            0.40,
            1,
            self.WHITE
        )

        action_name = (
            self.ACTION_NAMES[action]
            if action < len(
                self.ACTION_NAMES
            )
            else str(action)
        )

        self.draw_text(
            canvas,
            action_name,
            x + 95,
            y + 20,
            0.36,
            1,
            self.GREEN
        )

        self.draw_text(
            canvas,
            f"R:{float(reward):.1f}",
            x + width - 65,
            y + 20,
            0.34,
            1,
            self.WHITE
        )

        # -----------------------------------------------------
        # Frame
        # -----------------------------------------------------

        if frame is None:
            return

        frame = np.asarray(frame)

        if frame.ndim == 3:

            if frame.shape[2] == 3:

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR
                )

        elif frame.ndim == 2:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR
            )

        else:

            return

        image_w = width - 8
        image_h = height - 30

        frame = cv2.resize(
            frame,
            (
                image_w,
                image_h
            ),
            interpolation=cv2.INTER_NEAREST
        )

        canvas[
            y + 26:
            y + 26 + image_h,
            x + 4:
            x + 4 + image_w
        ] = frame

    # =========================================================
    # MARIO GRID
    # =========================================================

    def draw_mario_grid(
        self,
        canvas,
        frames,
        actions,
        rewards
    ):

        start_x = 15
        start_y = 55

        panel_w = 350
        panel_h = 150

        gap_x = 10
        gap_y = 10

        for i in range(
            min(
                self.num_envs,
                len(frames)
            )
        ):

            row = i // 2
            col = i % 2

            x = (
                start_x
                + col * (
                    panel_w
                    + gap_x
                )
            )

            y = (
                start_y
                + row * (
                    panel_h
                    + gap_y
                )
            )

            self.draw_mario_panel(
                canvas,
                frames[i],
                x,
                y,
                panel_w,
                panel_h,
                i,
                int(actions[i]),
                float(rewards[i])
            )

    # =========================================================
    # HUD
    # =========================================================

    def draw_hud(
        self,
        canvas,
        epsilon,
        step,
        loss,
        rewards
    ):

        y = 865

        avg_reward = 0.0

        if rewards is not None:

            avg_reward = float(
                np.mean(rewards)
            )

        values = [
            f"STEP  {step}",
            f"REWARD  {avg_reward:.2f}",
            f"LOSS  {loss if loss is not None else 0:.5f}",
            f"EPSILON  {epsilon:.4f}",
            f"ENVS  {self.num_envs}",
            "GPU  CUDA"
        ]

        x = 18

        for value in values:

            self.draw_text(
                canvas,
                value,
                x,
                y,
                0.42,
                1,
                (175, 195, 210)
            )

            x += 205

    # =========================================================
    # DRAW BRAIN
    # =========================================================

    def draw_brain(
        self,
        canvas,
        q_values,
        actions
    ):

        brain_x = 750

        # -----------------------------------------------------
        # Header
        # -----------------------------------------------------

        self.draw_text(
            canvas,
            "AI BRAIN",
            brain_x,
            35,
            0.82,
            2,
            self.WHITE
        )

        self.draw_text(
            canvas,
            "NEURAL SIGNAL FLOW",
            brain_x + 135,
            35,
            0.42,
            1,
            (140, 170, 190)
        )

        # -----------------------------------------------------
        # Layer positions
        # -----------------------------------------------------

        center_y = 280

        input_x = brain_x + 20
        conv1_x = brain_x + 125
        conv2_x = brain_x + 245
        conv3_x = brain_x + 365
        dense_x = brain_x + 485

        # -----------------------------------------------------
        # Layers
        # -----------------------------------------------------

        input_nodes = self.draw_input_layer(
            canvas,
            input_x,
            center_y
        )

        conv1_nodes = self.draw_layer(
            canvas,
            "CONV1",
            conv1_x,
            center_y,
            10,
            seed=10
        )

        conv2_nodes = self.draw_layer(
            canvas,
            "CONV2",
            conv2_x,
            center_y,
            10,
            seed=20
        )

        conv3_nodes = self.draw_layer(
            canvas,
            "CONV3",
            conv3_x,
            center_y,
            10,
            seed=30
        )

        dense_nodes = self.draw_dense_layer(
            canvas,
            dense_x,
            center_y
        )

        # -----------------------------------------------------
        # Connections
        # -----------------------------------------------------

        self.connect_layers(
            canvas,
            input_nodes,
            conv1_nodes,
            0.00
        )

        self.connect_layers(
            canvas,
            conv1_nodes,
            conv2_nodes,
            0.18
        )

        self.connect_layers(
            canvas,
            conv2_nodes,
            conv3_nodes,
            0.36
        )

        self.connect_layers(
            canvas,
            conv3_nodes,
            dense_nodes,
            0.54
        )

        # -----------------------------------------------------
        # Q signal
        # -----------------------------------------------------

        q_input = (
            dense_x + 10,
            center_y
        )

        q_output = (
            brain_x + 650,
            center_y
        )

        self.draw_signal(
            canvas,
            q_input,
            q_output,
            1.0,
            self.animation
        )

        self.draw_text(
            canvas,
            "Q",
            brain_x + 635,
            center_y - 25,
            0.60,
            2,
            self.GREEN
        )

        # -----------------------------------------------------
        # Q panel
        # -----------------------------------------------------

        self.draw_q_output(
            canvas,
            q_values,
            actions,
            brain_x,
            470
        )

        # -----------------------------------------------------
        # Architecture labels
        # -----------------------------------------------------

        labels = [
            ("4 x 84 x 84", input_x - 20),
            ("32 x 20 x 20", conv1_x - 25),
            ("64 x 9 x 9", conv2_x - 20),
            ("64 x 7 x 7", conv3_x - 20),
            ("512", dense_x - 10)
        ]

        for text, x in labels:

            self.draw_text(
                canvas,
                text,
                x,
                470,
                0.35,
                1,
                (130, 150, 165)
            )

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        frames,
        q_values,
        actions,
        epsilon,
        rewards,
        step,
        loss=None
    ):

        # -----------------------------------------------------
        # Canvas
        # -----------------------------------------------------

        canvas = np.zeros(
            (
                self.height,
                self.width,
                3
            ),
            dtype=np.uint8
        )

        canvas[:] = self.BLACK

        # -----------------------------------------------------
        # Title
        # -----------------------------------------------------

        self.draw_text(
            canvas,
            "SUPER MARIO",
            18,
            30,
            0.72,
            2,
            self.WHITE
        )

        self.draw_text(
            canvas,
            "DEEP Q-NETWORK",
            185,
            30,
            0.45,
            1,
            (140, 165, 185)
        )

        # -----------------------------------------------------
        # Mario
        # -----------------------------------------------------

        self.draw_mario_grid(
            canvas,
            frames,
            actions,
            rewards
        )

        # -----------------------------------------------------
        # Fake Brain
        # -----------------------------------------------------

        self.draw_brain(
            canvas,
            q_values,
            actions
        )

        # -----------------------------------------------------
        # HUD
        # -----------------------------------------------------

        self.draw_hud(
            canvas,
            epsilon,
            step,
            loss,
            rewards
        )

        # -----------------------------------------------------
        # Animation timing
        # -----------------------------------------------------

        current_time = time.time()

        dt = (
            current_time
            - self.last_time
        )

        self.last_time = current_time

        self.animation += (
            dt * 0.8
        )

        self.animation %= 1.0

        # -----------------------------------------------------
        # Display
        # -----------------------------------------------------

        cv2.imshow(
            self.window_name,
            canvas
        )

        key = cv2.waitKey(1)

        # ESC
        if key == 27:

            self.running = False
            cv2.destroyWindow(self.window_name)
            self.window_open = False

    def close(self):

        if self.window_open:
            cv2.destroyWindow(self.window_name)
            self.window_open = False

        self.running = False