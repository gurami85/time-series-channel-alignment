## 프롬프트
generate_virtual_data() 함수에서 채널 별 시계열을 생성하고(1), 노이즈를 추가하고(2), 채널을 무작위로 섞는(3) 전체 과정을 수학적으로 정의 및 전개해줘. 
- (1): Y[:, k] = np.sin(time_steps * (0.5 + 0.1k)) + np.cos(time_steps * (0.2k))
- (2): noisy_source = Y + np.random.normal(0, noise_level, (T, K))
- (3): perm = np.random.permutation(K)  # 예: [3, 15, 0, 1, ...]

## 응답
---
가상 데이터 생성 함수 `generate_virtual_data()` 과정의 수학적 정의 및 단계별 전개입니다.

---

### 1. 차원 및 변수 정의

* **시간 인덱스**: $t \in \{0, 1, \dots, T-1\}$ (총 타임스텝 수 $T$)
* **채널 인덱스**: $k \in \{0, 1, \dots, K-1\}$ (총 채널 수 $K$)
* **노드 인덱스**: $m \in \{1, 2, \dots, M\}$ (총 노드 수 $M$)
* **노이즈 표준편차**: $\sigma = \text{noise\_level}$

---

### 2. 단계별 수학적 전개

#### (1) 기준 시계열 행렬 $Y$ 생성 (Base Reference Signal Generation)

기준 시계열 행렬 $Y \in \mathbb{R}^{T \times K}$의 $t$번째 타임스텝, $k$번째 채널 성분 $Y_{t, k}$는 다음과 같이 정의됩니다.

$$Y_{t, k} = \sin\left(t \cdot (0.5 + 0.1k)\right) + \cos\left(t \cdot 0.2k\right)$$

행렬 $Y$ 전체 표현:


$$Y = \begin{bmatrix} Y_{0, 0} & Y_{0, 1} & \dots & Y_{0, K-1} \\ Y_{1, 0} & Y_{1, 1} & \dots & Y_{1, K-1} \\ \vdots & \vdots & \ddots & \vdots \\ Y_{T-1, 0} & Y_{T-1, 1} & \dots & Y_{T-1, K-1} \end{bmatrix}$$

---

#### (2) 노드별 가우시안 노이즈 첨가 (Node-wise Noise Addition)

각 노드 $m$마다 서로 독립적인 백색 가우시안 노이즈 행렬 $E^{(m)} \in \mathbb{R}^{T \times K}$를 생성합니다.

$$E_{t, k}^{(m)} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$$

노이즈가 포함된 노드 $m$의 기초 시계열 행렬 $\tilde{Y}^{(m)} \in \mathbb{R}^{T \times K}$는 다음과 같습니다.

$$\tilde{Y}^{(m)} = Y + E^{(m)}$$

성분별 표현:


$$\tilde{Y}_{t, k}^{(m)} = Y_{t, k} + E_{t, k}^{(m)} = \sin\left(t \cdot (0.5 + 0.1k)\right) + \cos\left(t \cdot 0.2k\right) + E_{t, k}^{(m)}$$

---

#### (3) 채널 무작위 순열 적용 (Channel Permutation)

각 노드 $m$마다 독립적인 전단사 순열 함수(Bijective Permutation Function) $\pi_m$을 정의합니다.

$$\pi_m : \{0, 1, \dots, K-1\} \to \{0, 1, \dots, K-1\}$$

여기서 $\pi_m(j)$는 노드 $m$의 $j$번째 관측 위치에 배치되는 원본 기준 시계열의 채널 인덱스를 의미합니다.

이를 순열 행렬(Permutation Matrix) $P^{(m)} \in {0, 1}^{K \times K}$로 표현하면 다음과 같습니다.

$$P_{k, j}^{(m)} = \begin{cases} 1, & \text{if } k = \pi_m(j) \\ 0, & \text{otherwise} \end{cases}$$

최종 관측 데이터 $X^{(m)} \in \mathbb{R}^{T \times K}$ (전체 텐서 $X \in \mathbb{R}^{M \times T \times K}$의 $m$번째 노드 슬라이스)는 순열 행렬과의 우곱(Right Multiplication)으로 전개됩니다.

$$X^{(m)} = \tilde{Y}^{(m)} P^{(m)} = (Y + E^{(m)}) P^{(m)}$$

성분별로 $t$번째 시간, $j$번째 관측 채널의 값 $X_{t, j}^{(m)}$는 다음과 같습니다.

$$X_{t, j}^{(m)} = \tilde{Y}_{t, \pi_m(j)}^{(m)} = Y_{t, \pi_m(j)} + E_{t, \pi_m(j)}^{(m)}$$

---

### 3. 최종 통합 수식 (Unified Formulation)

모든 단계를 하나로 결합한 노드 $m$, 타임스텝 $t$, 관측 채널 인덱스 $j$에서의 최종 데이터 $X_{m, t, j}$의 수식 표현입니다.

$$X_{m, t, j} = \sin\left(t \cdot \left(0.5 + 0.1\pi_m(j)\right)\right) + \cos\left(t \cdot 0.2\pi_m(j)\right) + \epsilon_{m, t, \pi_m(j)}$$

$$\text{where } \epsilon_{m, t, k} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$$

## 프롬프트
노이즈 레벨은 어떻게 표기하지?

## 응답
시계열 데이터 처리나 논문 작성 시 노이즈 레벨(Noise Level)을 수학적으로 표기하는 방식은 표현하고자 하는 목적(절대적 크기, 상대적 비율, 신호 대 잡음비)에 따라 크게 3가지로 나뉩니다.

---

### 1. 절대적 노이즈 크기: 표준편차 ($\sigma$) 또는 분산 ($\sigma^2$)

가우시안 백색 노이즈(Gaussian White Noise)를 추가할 때 가장 직관적으로 사용하며, 노이즈의 표준편차 자체를 노이즈 레벨로 정의합니다.

* **수식 표기**:

$$\epsilon \sim \mathcal{N}(0, \sigma^2)$$


* $\sigma$: 노이즈 레벨 (Noise Level / Standard Deviation)
* $\sigma^2$: 노이즈 전력 또는 분산 (Noise Variance)



---

### 2. 상대적 노이즈 비율: $\eta$ (Noise Ratio / Relative Intensity)

신호 $Y$의 크기(표준편차)에 비례하여 노이즈를 부여할 때 사용합니다. 신호 스케일과 무관하게 일정 비율의 노이즈를 섞고 싶을 때 유용합니다.

* **수식 표기**:

$$\sigma_{\text{noise}} = \eta \cdot \sigma_Y$$


* $\sigma_Y$: 원본 신호 $Y$의 표준편차 ($\text{std}(Y)$)
* $\eta$: 상대적 노이즈 레벨 (예: $\eta = 0.1$이면 신호 크기의 10% 수준 노이즈)


* **데이터 생성 수식**:

$$X = Y + \eta \cdot \sigma_Y \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$



---

### 3. 신호 대 잡음비: SNR (Signal-to-Noise Ratio)

신호 처리 및 통신 분야에서 노이즈의 영향력을 나타내는 표준 단위입니다.

* **선형 스케일 (Linear Scale)**:

$$\text{SNR} = \frac{P_{\text{signal}}}{P_{\text{noise}}} = \frac{\sigma_{\text{signal}}^2}{\sigma_{\text{noise}}^2}$$


* **데시벨 스케일 (dB Scale)**:

$$\text{SNR}_{\text{dB}} = 10 \log_{10} \left( \frac{P_{\text{signal}}}{P_{\text{noise}}} \right) = 20 \log_{10} \left( \frac{\sigma_{\text{signal}}}{\sigma_{\text{noise}}} \right)$$


* $\text{SNR}_{\text{dB}}$ 값이 높을수록 노이즈가 적은 깨끗한 신호이고, 낮을수록 노이즈 레벨이 높은 상태를 의미합니다.



---

### 추천 요약

* **파이썬 코드/실험 정의서**: `noise_level`을 가우시안 노이즈의 **표준편차 $\sigma$** 또는 상대 비율 $\eta$로 정의하고 명시합니다.
* **논문/보고서 수식**:
> *"가우시안 노이즈 $E \sim \mathcal{N}(0, \sigma^2)$를 추가하였으며, 여기서 $\sigma$는 노이즈 레벨을 나타낸다."*
> 또 또는
> *"신호 대 잡음비 $\text{SNR} = 10\text{ dB}$ 조건에서 실험을 진행하였다."*