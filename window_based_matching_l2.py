import cv2
import numpy as np


def distance_l2(x, y):
    return (x - y)**2


def window_based_matching_l2(left_img, right_img, disparity_range, kernel_size=5, save_result=True):
    # Read left and right images, then convert to grayscale
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[: 2]

    depth = np.zeros((height, width), np.uint8)

    kernel_half = int((kernel_size - 1) / 2)
    scale = 3
    max_value = 255 * 9

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            disparity = 0
            cost_min = 65534
            for j in range(disparity_range):
                total = 0
                value = 0
                for v in range(-kernel_half, kernel_half + 1):
                    for u in range(-kernel_half, kernel_half + 1):
                        value = max_value
                        if (x + u - j) >= 0:
                            value = distance_l2(
                                int(left[y + v, x + u]), int(right[y + v, (x + u) - j]))
                            total += value
                    if total < cost_min:
                        cost_min = total
                        disparity = j

                    # Let depth at (y, x) = j (disparity)
                    # Multiply by a scale factor for visualization purpose
                    depth[y, x] = disparity * scale

    if save_result == True:
        print("saving result...")
        # Save results
        cv2.imwrite(f"pixel_wise_11.png", depth)
        cv2.imwrite(f"pixel_wise_11_color.png",
                    cv2.applyColorMap(depth, cv2.COLORMAP_JET))

        print("Done.")

        return depth
