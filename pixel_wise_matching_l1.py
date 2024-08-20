import cv2
import numpy as np


def distance_l1(x, y):
    return abs(x - y)


def pixel_wise_matching_l1(left_img, right_img, disparity_range, save_result=True):
    # read left and right image, then convert to grayscale
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[: 2]

    # Create blank disparity map
    depth = np.zeros((height, width), np.uint8)
    scale = 16
    max_value = 255

    for y in range(height):
        for x in range(width):
            #  Find j where cost has minimum value
            disparity = 0
            cost_min = max_value

            for j in range(disparity_range):
                cost = max_value if (
                    x - j) < 0 else distance_l1(int(left[y, x]), int(right[y, x - j]))

                if cost < cost_min:
                    cost_min = cost
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
