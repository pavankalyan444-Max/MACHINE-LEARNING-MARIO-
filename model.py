import torch
import torch.nn as nn


class MarioCNN(nn.Module):

    def __init__(
        self,
        num_actions,
        input_channels=4,
        input_height=84,
        input_width=84
    ):

        super().__init__()

        # =========================================================
        # CNN FEATURE EXTRACTOR
        # =========================================================

        self.features = nn.Sequential(

            # -----------------------------------------------------
            # Layer 1
            # -----------------------------------------------------

            nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=8,
                stride=4
            ),

            nn.ReLU(),

            # -----------------------------------------------------
            # Layer 2
            # -----------------------------------------------------

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2
            ),

            nn.ReLU(),

            # -----------------------------------------------------
            # Layer 3
            # -----------------------------------------------------

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),

            nn.ReLU()
        )

        # =========================================================
        # AUTOMATICALLY CALCULATE CNN OUTPUT SIZE
        # =========================================================

        with torch.no_grad():

            dummy_input = torch.zeros(
                1,
                input_channels,
                input_height,
                input_width
            )

            dummy_output = self.features(
                dummy_input
            )

            feature_size = dummy_output.view(
                1,
                -1
            ).shape[1]

        # =========================================================
        # FULLY CONNECTED LAYERS
        # =========================================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                feature_size,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                num_actions
            )
        )

    # =============================================================
    # FORWARD PASS
    # =============================================================

    def forward(self, x):

        # ---------------------------------------------------------
        # Convert pixel values:
        #
        # 0 - 255
        #
        # into:
        #
        # 0.0 - 1.0
        # ---------------------------------------------------------

        if x.dtype == torch.uint8:

            x = x.float() / 255.0

        # ---------------------------------------------------------
        # CNN
        # ---------------------------------------------------------

        x = self.features(x)

        # ---------------------------------------------------------
        # Fully connected layers
        # ---------------------------------------------------------

        x = self.classifier(x)

        return x

