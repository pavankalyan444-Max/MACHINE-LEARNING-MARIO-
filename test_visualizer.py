import numpy as np

from brain_visualizer import BrainVisualizer


visualizer = BrainVisualizer()

frame = np.random.randint(
    0,
    256,
    (240, 256, 3),
    dtype=np.uint8
)

q_values = np.array([
    0.03,
    0.02,
    0.08,
    0.04,
    0.01,
    -0.02,
    0.00
])

while visualizer.running:

    visualizer.update(
        frames=[frame],
        q_values=np.expand_dims(q_values, axis=0),
        actions=np.array([int(np.argmax(q_values))]),
        epsilon=0.75,
        rewards=np.array([2.0]),
        step=100,
        loss=0.123456
    )

visualizer.close()