# main.py
# Combined (v2) from:
#   - lane_detector_hung.py
#   - MPC.py
#   - send_command.py
#
# Fixes for "steer almost constant":
# 1) MPC now ENFORCES initial state x0 = [ey, epsi, v] (before: x0 free -> weak response)
# 2) Corrected discrete kinematic model:
#    ey_{k+1} = ey_k + v*dt*epsi_k
#    epsi_{k+1} = epsi_k + v*dt/L * delta_k
#    v_{k+1} = v_k + dt * accel_k
# 3) Shows deg10 command (quantization check)

import time
import os
import cv2
import numpy as np

# -----------------------------
# Serial (from send_command.py)
# -----------------------------
try:
    import serial  # pip install pyserial
except ImportError:
    serial = None


class NucleoInterface:
    """BFMC serial protocol helper.

    BFMC expects commands terminated exactly as:  ';;\r\n'
      - #kl:30;;\r\n          (set power state / keep-alive)
      - #speed:<mm_s>;;\r\n   (signed int, -500..+500)
      - #steer:<deg10>;;\r\n  (signed int, -230..+230)
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baudrate: int = 115200,
        enable: bool = False,
        kl_value: int = 30,
        kl_period_s: float = 0.5,
    ):
        self.enable = bool(enable) and (serial is not None)
        self.ser = None
        self.kl_value = int(kl_value)
        self.kl_period_s = float(kl_period_s)
        self._last_kl_t = 0.0

        if self.enable:
            try:
                # write_timeout keeps the loop from blocking forever if USB is flaky
                self.ser = serial.Serial(
                    port=port,
                    baudrate=int(baudrate),
                    timeout=0.05,
                    write_timeout=0.05,
                )
                time.sleep(2.0)  # allow Nucleo to reboot
                try:
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                except Exception:
                    pass
                print(f"[Nucleo] Connected on {port}")
                # Many setups require kl to be sent once before speed/steer works.
                self.send_kl(self.kl_value, force=True)
            except Exception as e:
                print(f"[Nucleo] Cannot open serial: {e}")
                self.enable = False

    def _write(self, payload: str):
        if not (self.enable and self.ser is not None):
            return
        try:
            self.ser.write(payload.encode("ascii", errors="ignore"))
        except Exception as e:
            print(f"[Nucleo] Serial write error: {e}")
            self.enable = False

    def _ensure_kl(self):
        """Send #kl periodically so the powerboard stays active."""
        if not self.enable:
            return
        now = time.time()
        if (now - self._last_kl_t) >= self.kl_period_s:
            self.send_kl(self.kl_value, force=True)

    def send_kl(self, value: int = 30, force: bool = False):
        if not self.enable:
            return
        now = time.time()
        if (not force) and ((now - self._last_kl_t) < self.kl_period_s):
            return
        value = int(np.clip(int(value), 0, 30))
        self._write(f"#kl:{value};;\r\n")
        self._last_kl_t = now

    def send_steer_deg(self, steer_deg: float) -> int:
        """Send steering as deg, returns the actually sent deg10 (quantized + clipped)."""
        self._ensure_kl()
        deg10 = int(round(float(steer_deg) * 10.0))
        deg10 = int(np.clip(deg10, -230, 230))
        self._write(f"#steer:{deg10};;\r\n")
        return deg10

    def send_speed(self, speed_mm_s: int) -> int:
        """Send speed in mm/s, returns the actually sent value (clipped)."""
        self._ensure_kl()
        speed_mm_s = int(np.clip(int(speed_mm_s), -500, 500))
        self._write(f"#speed:{speed_mm_s};;\r\n")
        return speed_mm_s

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass


# -----------------------------
# MPC (OSQP)
# -----------------------------
import osqp
from scipy import sparse


class MPC:
    def __init__(self, dt=0.05, N=10, L=0.526):
        self.dt = float(dt)
        self.N = int(N)
        self.L = float(L)

        # Weight tuning
        self.Qy = 14.0       # lateral error (tuned up)
        self.Qpsi = 12.0     # heading error (tuned up)
        self.Qv = 1.0        # speed tracking
        self.Rd = 0.25       # steering effort (tuned down -> stronger steer)
        self.Ra = 0.20       # accel effort

        # --- BFMC protocol limits ---
        # steer: [-23deg, +23deg] (deg*10 => [-230, +230])
        # speed: [-500, +500] mm/s  ->  v in [-0.5, +0.5] m/s
        # For lane-following we typically drive forward only: v >= 0.
        self.v_ref = 0.40                 # m/s  (<= 0.50)
        self.delta_max = float(np.deg2rad(23.0))  # rad
        self.v_min = -0.50                # m/s (allow reverse)
        self.v_max =  0.50                 # m/s
        self.a_max = 1.0                  # m/s^2

        # --- Rate limits (recommended for BFMC servo stability) ---
        # These are NOT from the protocol table, but practical constraints to avoid jerk/oscillation.
        # You can tune them:
        #   ddelta_max_deg_per_step = 5..10 (deg / MPC step)
        #   da_max_per_step = 0.2..0.6 (m/s^2 / MPC step)
        self.ddelta_max = float(np.deg2rad(12.0))  # rad/step (allow faster steering change)
        self.da_max = 0.6                         # (m/s^2)/step

    def _get_dynamics(self, v):
        dt = self.dt
        L = self.L
        v = float(np.clip(v, self.v_min, self.v_max))
        v_eff = abs(v)  # use speed magnitude for linear model (stable in reverse)

        # x = [ey, epsi, v]
        # Small-angle linearized kinematic model:
        #   ey_{k+1}   = ey_k + v*dt*epsi_k
        #   epsi_{k+1} = epsi_k + v*dt/L * delta_k
        #   v_{k+1}    = v_k + dt * accel_k
        A = np.array([
            [1.0, v_eff * dt, 0.0],
            [0.0, 1.0,    0.0],
            [0.0, 0.0,    1.0]
        ], dtype=float)

        B = np.array([
            [0.0,         0.0],
            [v_eff * dt / L,  0.0],
            [0.0,         dt]
        ], dtype=float)

        return A, B

    def solve(self, ey, epsi, v, delta_prev=0.0, a_prev=0.0):
        """Return (delta_rad, accel, v_next). Adds BFMC limits + rate constraints."""
        N = self.N
        nx = 3
        nu = 2
        A, B = self._get_dynamics(v)

        # Decision vector z = [x0,u0, x1,u1, ..., x_{N-1}, u_{N-1}]
        nvar = N * (nx + nu)

        # Quadratic cost 0.5 z^T P z + q^T z
        P = np.zeros((nvar, nvar))
        q = np.zeros((nvar,))

        for k in range(N):
            base = k * (nx + nu)
            # x penalties: track ey->0, epsi->0, v->v_ref
            P[base + 0, base + 0] = 2 * self.Qy
            P[base + 1, base + 1] = 2 * self.Qpsi
            P[base + 2, base + 2] = 2 * self.Qv
            q[base + 2] = -2 * self.Qv * self.v_ref

            # u penalties
            P[base + 3, base + 3] = 2 * self.Rd
            P[base + 4, base + 4] = 2 * self.Ra

        P = sparse.csc_matrix(P)

        # -----------------------------
        # Constraints:
        # 1) variable bounds
        # 2) initial state x0 = [ey, epsi, v]
        # 3) initial input rate: u0 near (delta_prev, a_prev)
        # 4) dynamics: x_{k+1} = A x_k + B u_k
        # 5) input rate across horizon: u_k - u_{k-1}
        # -----------------------------
        # 1) Bounds (per stage: [ey, epsi, v, delta, accel])
        lb = []
        ub = []
        for _ in range(N):
            lb += [-np.inf, -np.inf, self.v_min, -self.delta_max, -self.a_max]
            ub += [ np.inf,  np.inf, self.v_max,  self.delta_max,  self.a_max]

        Abounds = sparse.eye(nvar, format="csc")

        # 2) Initial state
        Ainit = sparse.lil_matrix((nx, nvar))
        for i in range(nx):
            Ainit[i, i] = 1.0
        Ainit = Ainit.tocsc()
        x0 = np.array([float(ey), float(epsi), float(v)], dtype=float)
        linit = x0.copy()
        uinit = x0.copy()

        # 3) Initial input rate (u0 close to previous command)
        # u0 indexes within stage 0: delta at 3, accel at 4
        A_u0 = sparse.lil_matrix((nu, nvar))
        A_u0[0, 3] = 1.0
        A_u0[1, 4] = 1.0
        A_u0 = A_u0.tocsc()

        l_u0 = np.array([
            float(delta_prev) - self.ddelta_max,
            float(a_prev)     - self.da_max
        ], dtype=float)
        u_u0 = np.array([
            float(delta_prev) + self.ddelta_max,
            float(a_prev)     + self.da_max
        ], dtype=float)

        # 4) Dynamics constraints
        rows, cols, data = [], [], []
        ldyn, udyn = [], []
        for k in range(N - 1):
            xk = k * (nx + nu)
            uk = xk + nx
            xk1 = (k + 1) * (nx + nu)

            for r in range(nx):
                # x_{k+1}(r)
                rows.append(len(ldyn)); cols.append(xk1 + r); data.append(1.0)
                # -A*x_k
                for c in range(nx):
                    rows.append(len(ldyn)); cols.append(xk + c); data.append(-A[r, c])
                # -B*u_k
                for c in range(nu):
                    rows.append(len(ldyn)); cols.append(uk + c); data.append(-B[r, c])
                ldyn.append(0.0); udyn.append(0.0)

        Adyn = sparse.coo_matrix((data, (rows, cols)), shape=(len(ldyn), nvar)).tocsc()

        # 5) Input rate across horizon: (delta_k - delta_{k-1}), (a_k - a_{k-1})
        # For k=1..N-1
        rate_rows, rate_cols, rate_data = [], [], []
        lrate, urate = [], []
        for k in range(1, N):
            base_k = k * (nx + nu)
            base_km1 = (k - 1) * (nx + nu)

            # delta_k - delta_{k-1}
            r_idx = len(lrate)
            rate_rows += [r_idx, r_idx]
            rate_cols += [base_k + 3, base_km1 + 3]
            rate_data += [1.0, -1.0]
            lrate.append(-self.ddelta_max); urate.append(self.ddelta_max)

            # accel_k - accel_{k-1}
            r_idx = len(lrate)
            rate_rows += [r_idx, r_idx]
            rate_cols += [base_k + 4, base_km1 + 4]
            rate_data += [1.0, -1.0]
            lrate.append(-self.da_max); urate.append(self.da_max)

        Arate = sparse.coo_matrix((rate_data, (rate_rows, rate_cols)),
                                  shape=(len(lrate), nvar)).tocsc()

        # Stack all constraints
        Acons = sparse.vstack([Abounds, Ainit, A_u0, Adyn, Arate], format="csc")
        lcons = np.hstack([np.array(lb), linit, l_u0, np.array(ldyn), np.array(lrate)])
        ucons = np.hstack([np.array(ub), uinit, u_u0, np.array(udyn), np.array(urate)])

        prob = osqp.OSQP()
        prob.setup(P, q, Acons, lcons, ucons, verbose=False, warm_start=True)
        res = prob.solve()

        if res.info.status != "solved":
            return 0.0, 0.0, float(v)

        # stage 0: [ey0, epsi0, v0, delta0, accel0]
        delta0 = float(res.x[3])
        accel0 = float(res.x[4])
        v_next = float(v + accel0 * self.dt)
        v_next = float(np.clip(v_next, self.v_min, self.v_max))
        return delta0, accel0, v_next


# -----------------------------
# Lane detection pipeline (from lane_detector_hung.py)
# -----------------------------
def nothing(_=None):
    pass


def main():
    # ---- Video ----
    VIDEO_SOURCE = 0
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    # ---- Trackbars ----
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 130, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    # ---- ROI points (for 640x480) ----
    tl, bl = (37, 385), (7, 462)
    tr, br = (512, 391), (578, 461)

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    # ---- Sliding window params ----
    nwindows = 9
    margin = 50
    minpix = 50
    min_points = 120

    # ---- Conversions / refs ----
    ROAD_WIDTH_M = 0.40
    DEFAULT_SCALE_M_PER_PX = 0.0032
    y_ref = 479
    y_far = 300

    # ---- Controllers ----
    mpc = MPC(dt=0.05, N=10, L=0.526)
    # Serial port: Windows usually "COM3", Raspberry Pi typically "/dev/ttyACM0".
    default_port = "COM3" if os.name == "nt" else "/dev/ttyACM0"
    nucleo = NucleoInterface(port=default_port, baudrate=115200, enable=False)

    v_meas = 0.40
    last_delta = 0.0
    last_a = 0.0

    # --- Sign conventions ---
    # Your vehicle: steer < 0 => turn LEFT, steer > 0 => turn RIGHT.
    # The lane-derived errors (ey/epsi) may have opposite sign vs this convention,
    # so we apply STEER_SIGN to match your hardware.
    STEER_SIGN = 1.0
    STEER_GAIN = 1.10  # multiply steering output (<= keep within ±23° after clamp)

    # --- Smoothing (anti-oscillation) ---
    # Filter ey/epsi before MPC (reduces frame-to-frame noise)
    MEAS_LPF = 0.75      # 0.65..0.85 (higher = smoother, slower response)
    ey_f = 0.0
    epsi_f = 0.0

    # Smooth steering command before sending to Nucleo
    STEER_LPF_ALPHA = 0.80  # 0.70..0.90 (higher = smoother)
    MAX_STEP_DEG = 3.0      # max steering change per frame (deg), tune 2..6
    steer_cmd_prev = 0.0
    STRAIGHT_SETTLE = True
    STRAIGHT_EY = 0.010      # m
    STRAIGHT_EPSI = 0.018    # rad (~1 deg)
    # Estimate constant heading bias caused by BEV/ROI/fit (auto-calibration while driving straight)
    epsi_bias = 0.0
    BIAS_ALPHA = 0.08  # faster bias adaptation (0.05..0.15)
    EY_BIAS_GATE = 0.03  # learn bias only when near center (m)
    SLOPE_BIAS_GATE = 0.12  # learn bias only when lane is close to straight (|dx/dy|)
    EPSI_DEADBAND = 0.012  # rad (~1.1 deg)
    EY_DEADBAND = 0.004   # m (6 mm)
    prev_left_fit = None
    prev_right_fit = None

    last_t = time.time()
    fps = 0.0

    while True:
        ok, img = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # FPS
        now = time.time()
        dt = now - last_t
        if dt > 1e-6:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
        last_t = now

        frame = cv2.resize(img, (640, 480))

        # ROI points
        for p in [tl, bl, tr, br]:
            cv2.circle(frame, p, 5, (0, 0, 255), -1)

        # BEV warp
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(frame, M, (640, 480))

        # HSV threshold
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        lower = np.array([l_h, l_s, l_v], dtype=np.uint8)
        upper = np.array([u_h, u_s, u_v], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask2 = mask.copy()

        # Histogram base
        histogram = np.sum(mask[240:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        left_x = int(np.argmax(histogram[:midpoint])) if np.max(histogram[:midpoint]) > 0 else 150
        right_x = int(np.argmax(histogram[midpoint:]) + midpoint) if np.max(histogram[midpoint:]) > 0 else 490

        window_height = 480 // nwindows
        nonzero = mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_inds, right_inds = [], []
        mask_vis = mask.copy()

        for w in range(nwindows):
            y_low = 480 - (w + 1) * window_height
            y_high = 480 - w * window_height

            lx_low = left_x - margin
            lx_high = left_x + margin
            rx_low = right_x - margin
            rx_high = right_x + margin

            cv2.rectangle(mask_vis, (lx_low, y_low), (lx_high, y_high), 255, 2)
            cv2.rectangle(mask_vis, (rx_low, y_low), (rx_high, y_high), 255, 2)

            good_l = ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= lx_low) & (nonzerox < lx_high)).nonzero()[0]
            good_r = ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= rx_low) & (nonzerox < rx_high)).nonzero()[0]
            left_inds.append(good_l)
            right_inds.append(good_r)

            if len(good_l) > minpix:
                left_x = int(np.mean(nonzerox[good_l]))
            if len(good_r) > minpix:
                right_x = int(np.mean(nonzerox[good_r]))

        left_inds = np.concatenate(left_inds) if left_inds else np.array([], dtype=int)
        right_inds = np.concatenate(right_inds) if right_inds else np.array([], dtype=int)

        leftx = nonzerox[left_inds] if left_inds.size else np.array([])
        lefty = nonzeroy[left_inds] if left_inds.size else np.array([])
        rightx = nonzerox[right_inds] if right_inds.size else np.array([])
        righty = nonzeroy[right_inds] if right_inds.size else np.array([])

        result = frame.copy()
        ey_m, epsi = 0.0, 0.0

        left_fit, right_fit = None, None
        if len(leftx) > min_points:
            try:
                left_fit = np.polyfit(lefty, leftx, 2)
                prev_left_fit = left_fit
            except Exception:
                left_fit = None
        if len(rightx) > min_points:
            try:
                right_fit = np.polyfit(righty, rightx, 2)
                prev_right_fit = right_fit
            except Exception:
                right_fit = None
        if left_fit is None and prev_left_fit is not None:
            left_fit = prev_left_fit
        if right_fit is None and prev_right_fit is not None:
            right_fit = prev_right_fit

        if left_fit is not None and right_fit is not None:
            ploty = np.linspace(0, 479, 480)
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            lane_w_px = float(abs(right_fitx[-1] - left_fitx[-1]))
            m_per_px = ROAD_WIDTH_M / lane_w_px if lane_w_px > 1e-6 else DEFAULT_SCALE_M_PER_PX

            c_ref = 0.5 * (left_fitx[y_ref] + right_fitx[y_ref])
            c_far = 0.5 * (left_fitx[y_far] + right_fitx[y_far])

            img_center = 320.0
            ey_px = float(c_ref - img_center)
            ey_m = ey_px * m_per_px

            # Heading error from lane direction (use polynomial slope at y_ref)
            # x(y) = a*y^2 + b*y + c => dx/dy = 2a*y + b
            dy_dx_center = 0.5 * ((2*left_fit[0]*y_ref + left_fit[1]) + (2*right_fit[0]*y_ref + right_fit[1]))
            epsi = float(np.arctan(dy_dx_center))

            # fill lane in warp
            warp_zero = np.zeros_like(warped, dtype=np.uint8)
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(warp_zero, np.int32([pts]), (0, 255, 0))

            # unwarp overlay
            Minv = cv2.getPerspectiveTransform(pts2, pts1)
            newwarp = cv2.warpPerspective(warp_zero, Minv, (640, 480))
            result = cv2.addWeighted(frame, 1, newwarp, 0.5, 0)

        # ---- Preprocess errors (bias + deadband) ----
        # Learn heading bias when the lane is approximately straight.
        # This avoids "bias lock" where steering saturates and the previous steer-gate prevents learning.
        if 'dy_dx_center' in locals():
            if abs(ey_m) < EY_BIAS_GATE and abs(dy_dx_center) < SLOPE_BIAS_GATE and v_meas > 0.05:
                epsi_bias = (1.0 - BIAS_ALPHA) * epsi_bias + BIAS_ALPHA * epsi

        epsi_corr = float(epsi - epsi_bias)
        if abs(epsi_corr) < EPSI_DEADBAND:
            epsi_corr = 0.0
        if abs(ey_m) < EY_DEADBAND:
            ey_m = 0.0

        # Low-pass filter measurements used by MPC
        ey_f = MEAS_LPF * ey_f + (1.0 - MEAS_LPF) * float(ey_m)
        epsi_f = MEAS_LPF * epsi_f + (1.0 - MEAS_LPF) * float(epsi_corr)

# ---- MPC ----
        delta, accel_cmd, v_next = mpc.solve(ey_f, epsi_f, v_meas, delta_prev=last_delta, a_prev=last_a)
        # Match BFMC speed limit: ±500 mm/s => v in [0, 0.50] m/s for forward driving
        v_meas = float(np.clip(v_next, mpc.v_min, mpc.v_max))
        # store previous inputs for rate constraints in next iteration
        last_a = float(accel_cmd)
        last_delta = float(delta)

        # Convert to command units and clamp to BFMC limits
        steer_deg = float(np.rad2deg(delta))
        steer_deg *= STEER_SIGN
        steer_deg *= STEER_GAIN

        # If we are basically straight, gently pull the steering memory back to zero
        if STRAIGHT_SETTLE and abs(ey_f) < STRAIGHT_EY and abs(epsi_f) < STRAIGHT_EPSI:
            steer_cmd_prev *= 0.85

        # --- Steering smoothing (rate limit + low-pass) ---
        steer_deg_raw = steer_deg
        # rate limit to avoid oscillation/jitter
        step = float(np.clip(steer_deg_raw - steer_cmd_prev, -MAX_STEP_DEG, MAX_STEP_DEG))
        steer_deg_rl = steer_cmd_prev + step
        # low-pass filter for extra smoothness
        steer_deg = STEER_LPF_ALPHA * steer_cmd_prev + (1.0 - STEER_LPF_ALPHA) * steer_deg_rl
        steer_cmd_prev = steer_deg

        steer_deg = float(np.clip(steer_deg, -23.0, 23.0))
        speed_mm_s = int(round(v_meas * 1000.0))
        speed_mm_s = int(np.clip(speed_mm_s, -500, 500))

        # ---- Send ----
        nucleo.send_speed(speed_mm_s)
        deg10 = nucleo.send_steer_deg(steer_deg)

        # ---- Display ----
        cv2.putText(result, f"FPS: {fps:4.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(result, f"ey: {ey_m:+.3f} m", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(result, f"epsi: {epsi:+.3f} rad (bias {epsi_bias:+.3f})", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(result, f"delta(rad): {delta:+.3f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(result, f"steer: {steer_deg:+.1f} deg  (deg10={deg10:+d})", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(result, f"v: {v_meas:.2f} m/s", (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("Result", result)
        cv2.imshow("Bird's Eye View", warped)
        cv2.imshow("Sliding Windows", mask_vis)
        cv2.imshow("Threshold", mask2)

        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord("q"):
            break

    cap.release()
    nucleo.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
