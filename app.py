import pixel_wise_matching_l1
import pixel_wise_matching_l2
import window_based_matching_l1
import window_based_matching_l2

left_img_path = "/home/heigatvu/my-project/AIO2024/strereo-matching/data/tsukuba/left.png"
right_img_path = "/home/heigatvu/my-project/AIO2024/strereo-matching/data/tsukuba/right.png"
disparity_range = 16

left_img_path_2 = "/home/heigatvu/my-project/AIO2024/strereo-matching/data/Aloe/Aloe_left_1.png"
right_img_path_2 = "/home/heigatvu/my-project/AIO2024/strereo-matching/data/Aloe/Aloe_right_2.png"
disparity_range_2 = 64
kernel_size = 5

# pixel_wise_matching_l1.pixel_wise_matching_l1(
#     left_img_path, right_img_path, disparity_range, save_result=True)

# pixel_wise_matching_l2.pixel_wise_matching_l2(
#     left_img_path, right_img_path, disparity_range, save_result=True)

# window_based_matching_l2.window_based_matching_l2(
#     left_img_path_2, right_img_path_2, disparity_range_2, kernel_size, save_result=True)

# window_based_matching_l1.window_based_matching_l1(
#     left_img_path_2, right_img_path_2, disparity_range_2, kernel_size, save_result=True)


window_based_matching_l2.window_based_matching_l2(
    left_img_path_2, right_img_path_2, disparity_range_2, kernel_size, save_result=True)

window_based_matching_l1.window_based_matching_l1(
    left_img_path_2, right_img_path_2, disparity_range_2, kernel_size, save_result=True)
