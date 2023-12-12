import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import types

k_B = 1.380648e-23  # 볼츠만 상수 (J/K)

# 벡터의 크기를 계산하는 함수
def mod(v):
    return np.sum(v * v, axis=-1)

# 온도와 질량에 따른 속도 분포 함수
def pmod(v, T, m):
    return 4 * np.pi * v**2 * np.power(m / (2 * np.pi * k_B * T), 3 / 2) * np.exp(- m * v**2 / (2 * k_B * T))


class Simulation(animation.TimedAnimation):
    # 시뮬레이션의 초기 조건을 설정하는 초기화 메서드
    def __init__(self, n_particles, mass, rad, T, V, max_time, dt=0.2):
        self.PART = n_particles  # 입자 수
        self.MASS = mass         # 입자의 질량
        self.RAD = rad           # 입자의 반지름
        self.DIAM = 2 * rad      # 입자의 지름

        self.T = T               # 온도

        # 부피 함수 설정 (동적 또는 정적)
        if isinstance(V, types.FunctionType):
            self.V0 = V(0)
            self.V = V
            self.Vconst = False
        else:
            self.V0 = V
            self.V = lambda t: V
            self.Vconst = True

        self.L = np.power(self.V0, 1/3)    # 상자의 한 변의 길이
        self.halfL = self.L / 2            # 상자의 중심까지의 거리
        self.A = 6 * self.L**2             # 상자의 전체 표면적

        self.max_time = max_time  # 최대 시간
        self.dt = dt              # 시간 간격
        self.Nt = int(max_time / self.dt)  # 시간 단계 수

        self.evaluate_properties()  # 물리적 성질 계산

        # 속도 히스토그램 설정
        self.min_v = 0
        self.max_v = self.vmax * 3
        self.dv = 0.2  # 속도 구간 (m/s)
        self.Nv = int((self.max_v - self.min_v) / self.dv)

        # 압력 계산 설정
        self.dP = 1  # 압력 계산 간격 (s)
        self.NP = int(max_time / self.dP)  # 압력 계산 단계 수

        self.init_particles()  # 입자 초기화
        self.init_figures()    # 그래픽 초기화

        animation.TimedAnimation.__init__(self, self.fig, interval=1, blit=True, repeat=False)

    # 시뮬레이션의 물리적 성질을 계산하는 메서드
    def evaluate_properties(self):
        self.P = self.PART * k_B * self.T / self.V0      # 압력
        self.U = 1.5 * self.PART * k_B * self.T          # 내부 에너지
        self.vrms = np.sqrt(3 * k_B * self.T / self.MASS)  # 평균 제곱근 속도
        self.vmax = np.sqrt(2 * k_B * self.T / self.MASS)  # 최대 속도
        self.vmed = np.sqrt(8 * k_B * self.T / (np.pi * self.MASS))  # 중간 속도

    # 입자의 초기 위치와 속도를 설정하는 메서드
    def init_particles(self):
        # 입자의 초기 위치 무작위 설정
        self.r = np.random.rand(self.PART, 3) * 2 * (self.halfL - self.RAD) - (self.halfL - self.RAD)

        v_polar = np.random.random((self.PART, 2))  # 무작위 속도 분포

        self.v = np.zeros((self.PART, 3))  # 속도 초기화
        
        # 구면 좌표계에서의 속도 벡터 계산
        self.v[:, 0] = np.sin(v_polar[:, 0] * np.pi) * np.cos(v_polar[:, 1] * 2 * np.pi)
        self.v[:, 1] = np.sin(v_polar[:, 0] * np.pi) * np.sin(v_polar[:, 1] * 2 * np.pi)
        self.v[:, 2] = np.cos(v_polar[:, 0] * np.pi)

        self.v *= self.vrms  # 속도 벡터에 평균 제곱근 속도 적용

    # 시뮬레이션의 그래픽을 초기화하는 메서드
    def init_figures(self):
        self.fig = plt.figure()  # 그림 객체 생성

        self.ax1 = self.fig.add_subplot(111, projection='3d')  # 3D 그래프 준비

        # 입자 및 질량 중심의 위치 표시를 위한 준비
        self.line_3d = self.ax1.plot([], [], [], ls='None', marker='.')[0]
        self.line_3d_cm = self.ax1.plot([0], [0], [0], ls='None', marker='.', color='r')[0]

        self.line_xy = self.ax1.plot([], [], ls='None', marker='.')[0]
        self.line_xy_cm = self.ax1.plot([0], [0], ls='None', marker='.', color='r')[0]

        self.line_yz = self.ax1.plot([], [], ls='None', marker='.')[0]
        self.line_yz_cm = self.ax1.plot([0], [0], ls='None', marker='.', color='r')[0]

        self.line_xz = self.ax1.plot([], [], ls='None', marker='.')[0]
        self.line_xz_cm = self.ax1.plot([0], [0], ls='None', marker='.', color='r')[0]

        # 속도 히스토그램 준비
        self.vel_x = np.linspace(self.min_v, self.max_v, self.Nv)
        self.vel_y = np.zeros(self.Nv)

        self.line_vel = self.ax1.plot([], [], color='b', lw=0.5)[0]

        # 압력 계산을 위한 준비
        self.ex_p = 0.0  # 벽과의 충돌로 인해 교환된 운동량
        self.last_P = -1
        self.P_x = np.zeros(self.NP)
        self.P_y = np.zeros(self.NP)

        self.line_p = self.ax1.plot([], [], color='b', lw=0.5)[0]

        self._drawn_artists = [  # 그래픽 요소 목록
            self.line_3d, self.line_3d_cm,
            self.line_xy, self.line_xy_cm,
            self.line_yz, self.line_yz_cm,
            self.line_xz, self.line_xz_cm,
            self.line_vel, self.line_p]

    # 시뮬레이션 볼륨(부피)을 업데이트하는 메서드
    def update_volume(self, t):

        self.V0 = self.V(t)
        self.L = np.power(self.V0, 1/3)
        self.halfL = self.L / 2
        self.A = 6 * self.L**2

        box_limits = [-self.halfL, self.halfL]
        self.ax1.set_xlim3d(box_limits)
        self.ax1.set_ylim3d(box_limits)
        self.ax1.set_zlim3d(box_limits)

    # 각 프레임마다 입자들의 위치를 업데이트하고 그래픽을 그리는 메서드
    def _draw_frame(self, t):
        self.update_volume(t)  # 부피 업데이트

        # 입자 위치 업데이트
        self.r += self.dt * self.v

        # 다른 입자들과의 충돌 확인
        dists = np.sqrt(mod(self.r - self.r[:, np.newaxis]))
        cols2 = (0 < dists) & (dists < self.DIAM)
        idx_i, idx_j = np.nonzero(cols2)

        for i, j in zip(idx_i, idx_j):
            if j < i:
                # 중복 및 동일 입자 건너뛰기
                continue

            rij = self.r[i] - self.r[j]
            d = mod(rij)
            vij = self.v[i] - self.v[j]
            dv = np.dot(vij, rij) * rij / d
            self.v[i] -= dv
            self.v[j] += dv

            # 접촉하지 않도록 위치 업데이트
            self.r[i] += self.dt * self.v[i]
            self.r[j] += self.dt * self.v[j]

        # 벽과의 충돌 확인
        walls = np.nonzero(np.abs(self.r) + self.RAD > self.halfL)
        self.v[walls] *= -1
        self.r[walls] -= self.RAD * np.sign(self.r[walls])

        # 질량 중심의 위치 계산
        CM = np.sum(self.r, axis=0) / self.PART

        # 새로운 좌표 플롯
        self.line_3d.set_data(self.r[:, 0], self.r[:, 1])
        self.line_3d.set_3d_properties(self.r[:, 2])

        self.line_3d_cm.set_data(CM[0], CM[1])
        self.line_3d_cm.set_3d_properties(CM[2])

        self.line_xy.set_data(self.r[:, 0], self.r[:, 1])
        self.line_xy_cm.set_data(CM[0], CM[1])

        self.line_yz.set_data(self.r[:, 1], self.r[:, 2])
        self.line_yz_cm.set_data(CM[1], CM[2])

        self.line_xz.set_data(self.r[:, 0], self.r[:, 2])
        self.line_xz_cm.set_data(CM[0], CM[2])

        # 속도의 히스토그램 생성
        v_mod = np.sqrt(mod(self.v))

        for k in range(self.Nv):
            self.vel_y[k] = np.count_nonzero((k*self.dv < v_mod) & (v_mod < (k + 1)*self.dv))

        self.line_vel.set_data(self.vel_x, self.vel_y)

        # 이번 반복에서 교환된 운동량을 누적된 것에 추가
        self.ex_p += 2 * self.MASS * np.sum(np.abs(self.v[walls]))
        i = int(t / self.dP)
        if i > self.last_P + 1:
            # self.dP seconds 후의 압력 계산

            self.last_P = i - 1

            A_avg = self.A if self.Vconst else (self.A + 6 * np.power(self.V(t - self.dP), 2/3)) / 2

            self.P_x[self.last_P] = (t if self.Vconst else self.V0)
            self.P_y[self.last_P] = self.ex_p / (self.dP * A_avg)

            self.ex_p = 0.0

            self.line_p.set_data(self.P_x[:i], self.P_y[:i])

    # 애니메이션 프레임 시퀀스 생성
    def new_frame_seq(self):
        return iter(np.linspace(0, self.max_time, self.Nt))

# 부피 변화 함수
def V(t, V0, Vf, t_max):
    return V0 + (Vf - V0) * t / t_max

# 시뮬레이션 매개변수 설정
PARTICLES = 200       # 입자 수
MASS = 1.2e-20        # 입자의 질량
RADIUS = 0.01         # 입자의 반지름
TEMPERATURE = 20      # 온도
V0, Vf = 0.5, 15      # 초기 및 최종 부피
T_MAX = 1000          # 최대 시뮬레이션 시간

# 시뮬레이션 실행 및 애니메이션 표시
ani = Simulation(PARTICLES, MASS, RADIUS, TEMPERATURE, 2, T_MAX, 0.05)
plt.show()