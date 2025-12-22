import cv2
import numpy as np

vidcap = cv2.VideoCapture("Xin.mp4")


def nothing(x):
    pass


cv2.namedWindow("Trackbars")
# Tăng giới hạn mặc định của U-S lên 255 để không bị mất màu
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 130, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Khởi tạo biến lưu trạng thái cũ
prev_left_fit = None
prev_right_fit = None

while True:
    ret, image = vidcap.read()
    if not ret:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(image, (640, 480))

    #ROI & WARP
    tl, bl = (37, 385), (7, 462)
    tr, br = (512, 391), (578, 461)

    # Vẽ điểm đỏ để dễ chỉnh (Debug)
    cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    cv2.circle(frame, br, 5, (0, 0, 255), -1)

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(frame, matrix, (640, 480))

    #THRESHOLD
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower, upper)
    mask2 = mask.copy()
    #HISTOGRAM & SLIDING WINDOW
    histogram = np.sum(mask[240:, :], axis=0)
    midpoint = histogram.shape[0] // 2

    # Tìm điểm khởi đầu
    left_x_current = np.argmax(histogram[:midpoint]) if np.max(histogram[:midpoint]) > 0 else 150
    right_x_current = np.argmax(histogram[midpoint:]) + midpoint if np.max(histogram[midpoint:]) > 0 else 490

    nwindows = 9
    window_height = 480 // nwindows

    # Các vị trí pixel khác 0 trong ảnh
    nonzero = mask.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = []
    right_lane_inds = []

    margin = 50
    minpix = 50

    msk = np.dstack((mask, mask, mask)) * 255  # Để vẽ màu lên mask

    for window in range(nwindows):
        # Xác định biên cửa sổ
        win_y_low = 480 - (window + 1) * window_height
        win_y_high = 480 - window * window_height

        # Biên trái
        win_xleft_low = left_x_current - margin
        win_xleft_high = left_x_current + margin
        # Biên phải
        win_xright_low = right_x_current - margin
        win_xright_high = right_x_current + margin

        # Vẽ hình chữ nhật để visualize
        cv2.rectangle(mask, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 255, 255), 2)
        cv2.rectangle(mask, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 255, 255), 2)

        # Tìm các pixel nằm trong cửa sổ
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Logic cập nhật tâm cửa sổ (Sliding)
        if len(good_left_inds) > minpix:
            left_x_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_x_current = int(np.mean(nonzerox[good_right_inds]))

    # Gộp các chỉ số lại
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Trích xuất tọa độ pixel của làn đường
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    result = frame.copy()

    # POLYFIT & CALCULATION
    # Chỉ fit nếu tìm thấy đủ điểm
    if len(leftx) > 100 and len(rightx) > 100:
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            prev_left_fit = left_fit
            prev_right_fit = right_fit

            # Tạo dữ liệu y để vẽ
            ploty = np.linspace(0, 479, 480)

            # Tính x từ phương trình bậc 2: x = ay^2 + by + c
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            # --- TÍNH TOÁN ---
            # 1. Độ cong (tại đáy ảnh y=480)
            y_eval = 480
            left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
            right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(
                2 * right_fit[0])
            curvature = (left_curverad + right_curverad) / 2

            # 2. Offset
            lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
            image_center = 320  # 640/2
            offset = (image_center - lane_center + 0.15) * (3.7 / 700)  # Giả sử độ rộng làn pixel là 700

            # 3. Steering Angle
            # Lấy điểm mục tiêu ở giữa ảnh (hoặc xa hơn chút)
            look_ahead_idx = 400  # Giữa ảnh
            target_x = (left_fitx[look_ahead_idx] + right_fitx[look_ahead_idx]) / 2
            target_y = look_ahead_idx  # theo coordinate gốc của warped (y chạy từ 0-480) -> thực tế đây là y trong warped

            # Chuyển đổi tọa độ để tính góc (gốc tọa độ ở giữa đáy xe)
            dx = target_x - 320
            dy = 480 - target_y  # Khoảng cách theo y từ đáy lên điểm target
            steering = np.degrees(np.arctan(dx / dy))

            # --- VẼ ---
            # Vẽ vùng làn đường màu xanh lá cây đè lên ảnh gốc (Unwarp)
            # Tạo một hình ảnh trắng để vẽ làn
            warp_zero = np.zeros_like(warped).astype(np.uint8)

            # Gom điểm để vẽ đa giác
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            cv2.fillPoly(warp_zero, np.int_([pts]), (0, 255, 0))

            # Unwarp (đưa về góc nhìn gốc)
            Minv = cv2.getPerspectiveTransform(pts2, pts1)
            newwarp = cv2.warpPerspective(warp_zero, Minv, (640, 480))

            # Gộp vào ảnh gốc
            result = cv2.addWeighted(frame, 1, newwarp, 0.5, 0)

            # Vẽ thông tin
            cv2.line(result, (320, 480), (int(target_x), int(480 - dy)), (0, 0, 255),
                     3)  # Line chỉ hướng (tương đối vì target_x là trên warped)
            cv2.putText(result, f"Steer: {steering:.2f} deg", (320, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(result, f"Offset: {offset:.2f} m", (320, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        except Exception as e:
            pass  # Nếu lỗi fit thì bỏ qua frame này

    cv2.imshow("Result", result)
    cv2.imshow("Bird's Eyes View", warped)
    cv2.imshow("Sliding Windows", mask)
    cv2.imshow("hihi", mask2)
    if cv2.waitKey(10) == 27:
        break

vidcap.release()
cv2.destroyAllWindows()