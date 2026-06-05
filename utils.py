import numpy as np


def contrast_power_law(image: np.ndarray, constant: float = 1, gamma: float = 0.5):
    norm = image / 255.0
    enhanced = (constant * np.power(norm, gamma)) * 255
    return np.asarray(enhanced).astype(np.uint8)

if __name__ == "__main__":
    res = contrast_power_law(np.zeros((200, 200)))
    print(res)