## 프롬프트
다시 이 연구를 구성하는 주요 개념부터 확실히 짚고 넘어가려고 해. 이 논문은 범용적으로 어떤 수리적인 문제를 다루는거지?

## 응답
본 연구가 다루는 범용적 수학 문제의 본질을 핵심 개념 위주로 깔끔하게 정리해 드립니다.

### 📌 핵심 문제 정의

본 논문은 "다중 에이전트 다변량 시계열의 역순열 조합 최적화 문제(Combinatorial Optimization for Inverse Permutation of Multi-Agent Multivariate Time-Series)"를 다룹니다.

이 문제는 신호 처리 및 최적화 학계에서 전형적인 **NP-hard** 문제로 분류됩니다.

---

### 1. 수학적 수식 모델링

* **기준 시계열 (Reference, $Y$):** $T$개의 시간 축과 $K$개의 채널을 가진 기준 데이터 $Y \in \mathbb{R}^{T \times K}$
* **소스 시계열 (Sources, $X$):** $M$개의 소스 노드 각각이 $K$개의 채널을 가지는 데이터 $X_m \in \mathbb{R}^{T \times K} \quad (m = 1, 2, \dots, M)$
* **채널 섞임 (Shuffling Matrix, $P_m$):** 각 소스 $m$은 알 수 없는 $K \times K$ 크기의 순열 행렬 $P_m$에 의해 채널 순서가 뒤섞인 상태입니다.
* **복원 및 집계 (Reconstruction & Aggregation):**
후보 순열 $\hat{P}_m$을 적용하여 정렬한 뒤, $M$개 소스의 평균을 내어 추정 시계열 $\hat{Y}$를 생성합니다.

$$\hat{Y} = \frac{1}{M} \sum_{m=1}^{M} X_m \hat{P}_m^{-1}$$



---

### 2. 목적 함수 (Objective Function)

추정 시계열 $\hat{Y}$와 기준 시계열 $Y$ 간의 **채널별 평균 피어슨 상관계수(Pearson Correlation Coefficient)를 극대화**하는 순열 조합 $(\hat{P}_1, \hat{P}_2, \dots, \hat{P}_M)$을 찾는 것입니다.

$$\max_{\hat{P}_1, \dots, \hat{P}_M} \frac{1}{K} \sum_{k=1}^{K} \text{Corr}\left(Y_{:, k}, \hat{Y}_{:, k}\right)$$

---

### 3. 본 문제가 수학적으로 대단히 어려운 이유

1. **조합 폭발 (Combinatorial Explosion):**
* 단일 소스의 가능한 경우의 수: $K!$
* 전체 $M$개 소스의 전체 탐색 공간: $(K!)^M$
* $K=16, M=30$ 조건에서는 경우의 수가 $(20\text{조})^{30}$으로, 전수 조사가 물리적으로 불가능합니다.


2. **국소 최적해 함정 (Local Optima & Non-convexity):**
* $M$개 소스가 서로 얽혀서 평균($\hat{Y}$)을 이루기 때문에, 특정 소스 하나만 맞춘다고 해서 상관계수 결과가 정직하게 비례하여 상승하지 않습니다.
* 이에 따라 경사 하강법이나 헝가리안 알고리즘 같은 탐욕적(Greedy) 기법은 극초반 함정에 빠져 수렴하지 못합니다.



---

### 💡 한 줄 요약

> **"순서가 꼬이고 노이즈가 섞인 $M$개의 센서 데이터 뭉치들을, 단 하나의 기준 파형 $Y$만 보고 역순열을 추정하여 원래의 채널 위치로 완벽히 복원해 내는 조합 최적화 문제"**

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

---
## 프롬프트
채널 정렬을 위한 순열 최적화 과정을 수학적으로 전개해줘.

## 응답
$M$개 노드의 뒤섞인 시계열 채널을 평균 집계(Aggregated Mean) 기반으로 정렬하는 **순열 최적화(Permutation Optimization)** 문제의 수학적 전개 과정입니다.

---

### 1. 의사결정 변수 및 탐색 공간 정의 (Decision Variables & Search Space)

* **대칭군(Symmetric Group) 정의**: $K$개 채널의 가능한 모든 순열의 집합을 대칭군 $\mathcal{S}_K$라 하고, 그 크기는 $\vert{}\mathcal{S}_K\vert{} = K!$ 입니다.
* **노드별 후보 순열 (Candidate Permutation)**: 노드 $m$ ($1 \le m \le M$)에 할당되는 채널 정렬 함수를 $p_m \in \mathcal{S}_K$로 정의합니다.

$$p_m = [p_{m, 0}, p_{m, 1}, \dots, p_{m, K-1}]$$



여기서 $p_{m, j}$는 노드 $m$의 $j$번째 관측 채널이 대응하는 원본 채널의 인덱스를 의미합니다.
* **역순열 (Inverse Permutation)**: 원본 $k$번째 채널 복원을 위해 관측 데이터에서 불러올 채널 인덱스를 가리키는 역순열 매핑 함수 $q_m \in \mathcal{S}_K$를 구합니다.

$$q_m = \text{argsort}(p_m) \implies q_{m, k} = p_m^{-1}(k)$$


* **전체 결정 변수 조합 공간 ($\mathbf{P}$)**: $M$개 노드의 순열 집합 $\mathbf{P}$의 전체 탐색 공간 크기는 다음과 같습니다.

$$\mathbf{P} = \{p_1, p_2, \dots, p_M\} \in (\mathcal{S}_K)^M, \quad \vert{}\mathbf{P}\vert{} = (K!)^M$$



---

### 2. 채널 재구성 및 평균 집계 연산자 (Reconstruction & Aggregation)

후보 순열 조합 $\mathbf{P}$가 주어졌을 때, 각 노드 $m$의 관측 데이터 $X^{(m)} \in \mathbb{R}^{T \times K}$로부터 복원된 시계열 $\hat{X}^{(m)}(p_m) \in \mathbb{R}^{T \times K}$의 $t$번째 타임스텝, $k$번째 채널 성분은 다음과 같습니다.

$$\hat{X}_{t, k}^{(m)}(p_m) = X_{t, q_{m, k}}^{(m)}$$

모든 $M$개 노드의 복원 시계열을 평균 산술 집계한 시계열 행렬 $\bar{X}(\mathbf{P}) \in \mathbb{R}^{T \times K}$의 $k$번째 채널 성분 $\bar{X}_{:, k}(\mathbf{P}) \in \mathbb{R}^T$는 다음과 같이 정의됩니다.

$$\bar{X}_{t, k}(\mathbf{P}) = \frac{1}{M} \sum_{m=1}^{M} \hat{X}_{t, k}^{(m)}(p_m) = \frac{1}{M} \sum_{m=1}^{M} X_{t, q_{m, k}}^{(m)}$$

---

### 3. 목적함수 정의 (Objective Function)

평균 집계된 시계열 $\bar{X}_{:, k}(\mathbf{P})$와 타겟 시계열 $Y_{:, k}$ 간의 유사도를 측정하기 위해 피어슨 상관계수(Pearson Correlation Coefficient) 연산자 $\rho(\cdot, \cdot)$를 사용합니다.

#### 채널별 피어슨 상관계수

$$\rho\left(\bar{X}_{:, k}(\mathbf{P}), Y_{:, k}\right) = \frac{\sum_{t=0}^{T-1} \left(\bar{X}_{t, k}(\mathbf{P}) - \mu_{\bar{X}, k}\right) \left(Y_{t, k} - \mu_{Y, k}\right)}{\sqrt{\sum_{t=0}^{T-1} \left(\bar{X}_{t, k}(\mathbf{P}) - \mu_{\bar{X}, k}\right)^2} \cdot \sqrt{\sum_{t=0}^{T-1} \left(Y_{t, k} - \mu_{Y, k}\right)^2}}$$

여기서 평균값은 다음과 같습니다.


$$\mu_{\bar{X}, k} = \frac{1}{T} \sum_{t=0}^{T-1} \bar{X}_{t, k}(\mathbf{P}), \quad \mu_{Y, k} = \frac{1}{T} \sum_{t=0}^{T-1} Y_{t, k}$$

#### 전체 채널 평균 적응도 함수 $J(\mathbf{P})$

모든 $K$개 채널에 대한 상관계수의 평균을 최적화의 목적함수로 설정합니다.

$$J(\mathbf{P}) = \frac{1}{K} \sum_{k=0}^{K-1} \rho\left(\bar{X}_{:, k}(\mathbf{P}), Y_{:, k}\right)$$

---

### 4. 최적화 문제의 최종 수식화 (Optimization Problem Formulation)

이 정렬 문제는 탐색 공간 $(\mathcal{S}_K)^M$ 상에서 목적함수 $J(\mathbf{P})$를 최대화하는 조합 최적화 문제(Combinatorial Optimization Problem)로 귀결됩니다.

$$\mathbf{P}^* = \arg\max_{\mathbf{P} \in (\mathcal{S}_K)^M} J(\mathbf{P})$$

$$\text{subject to } p_m \in \mathcal{S}_K \quad (\forall m \in \{1, 2, \dots, M\})$$

---

### 5. 노이즈 레벨 $\sigma$에 따른 통계적 영향 및 평균 집계의 효과

첨부된 이미지의 (b), (c), (d)와 같이 노이즈 레벨이 $\sigma = 0.1 \to 0.5 \to 0.9$로 증가하는 경우, 관측 신호 $X_{t, j}^{(m)} = Y_{t, \pi_m(j)} + E_{t, \pi_m(j)}^{(m)}$의 신호 대 잡음비(SNR)가 급격히 저하됩니다.

올바른 정렬 순열 $\mathbf{P}^*$를 찾았을 때, 평균 집계 시계열 $\bar{X}_{t, k}(\mathbf{P}^*)$에 잔류하는 노이즈 성분은 다음과 같습니다.

$$\bar{E}_{t, k} = \frac{1}{M} \sum_{m=1}^{M} E_{t, k}^{(m)}$$

각 노드의 노이즈가 독립동일분포 $E_{t, k}^{(m)} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$를 따른다고 가정하면, 집계된 노이즈의 분산 및 표준편차는 다음과 같이 감쇄됩니다.

$$\operatorname{Var}(\bar{E}_{t, k}) = \frac{\sigma^2}{M} \implies \operatorname{Std}(\bar{E}_{t, k}) = \frac{\sigma}{\sqrt{M}}$$

따라서 노드 수 $M$이 증가할수록 집계 시계열의 유효 노이즈 레벨이 $\frac{1}{\sqrt{M}}$ 배로 줄어들므로, (d)와 같이 $\sigma = 0.9$ 수준의 극심한 노이즈 조건에서도 평균 집계 기반 목적함수 $J(\mathbf{P})$가 강건한(Robust) 강한 상관계수 피크를 형성할 수 있게 됩니다.

---
## 프롬프트

첨부된 이미지는 너가 생성한 plot_permutation_comparison() 함수의 시각화 결과야. 이미지를 보면 관측 시계열 X의 채널이 뒤섞인 상태에서 평균 집계한 시계열을 Raw Mean X 라고 표시되었고, 채널 정렬 후 평균 집계한 시계열을 Aligned Mean X 라고 표시되었어. 너가 위에서 정리해준 수학적 정의 및 전개 과정을 토대로 두 용어를 수학적으로 표기해줘.


## 응답

제시해주신 시각화 결과 그래프의 `Raw Mean X`와 `Aligned Mean X`를 앞에서 정립한 수학적 정의에 맞춰 체계적으로 정리한 표기법입니다.

---

### 1. Raw Mean X (정렬 전 단순 평균 시계열)

채널 뒤섞임(Permutation)을 복원하지 않은 상태, 즉 **각 노드의 관측 채널 순서 그대로 $M$개 노드를 단순 산술 평균**한 시계열입니다. (항등 순열 $I$를 적용한 상태와 동일합니다.)

#### ① 타임스텝 $t$, 채널 $k$ 성분 표기

$$\bar{X}_{t, k}^{\text{raw}} = \frac{1}{M} \sum_{m=1}^{M} X_{t, k}^{(m)}$$

#### ② $k$번째 채널 시계열 벡터 표기 ($T \times 1$)

$$\bar{X}_{:, k}^{\text{raw}} = \frac{1}{M} \sum_{m=1}^{M} X_{:, k}^{(m)}$$

#### ③ 전체 시계열 행렬 표기 ($T \times K$)

$$\bar{X}^{\text{raw}} = \frac{1}{M} \sum_{m=1}^{M} X^{(m)}$$

---

### 2. Aligned Mean X (채널 정렬 후 집계 평균 시계열)

최적화 알고리즘을 통해 추정된 **최적 순열 조합 $\mathbf{P}^* = {p_1^*, p_2^*, \dots, p_M^*}$의 역순열 매핑 $q_{m, k}^* = (p_m^*)^{-1}(k)$를 각 노드에 적용하여 복원한 뒤 $M$개 노드를 산술 평균**한 시계열입니다.

#### ① 타임스텝 $t$, 채널 $k$ 성분 표기

$$\bar{X}_{t, k}^{\text{aligned}} = \bar{X}_{t, k}(\mathbf{P}^*) = \frac{1}{M} \sum_{m=1}^{M} X_{t, q_{m, k}^*}^{(m)}$$

* 여기서 $q_{m, k}^*$는 노드 $m$의 관측 데이터 $X^{(m)}$에서 원본 $k$번째 채널 위치에 해당하도록 재배열할 관측 채널 인덱스입니다.

#### ② $k$번째 채널 시계열 벡터 표기 ($T \times 1$)

$$\bar{X}_{:, k}^{\text{aligned}} = \bar{X}_{:, k}(\mathbf{P}^*) = \frac{1}{M} \sum_{m=1}^{M} X_{:, q_{m, k}^*}^{(m)}$$

#### ③ 전체 복원 시계열 행렬 표기 ($T \times K$)

$$\bar{X}^{\text{aligned}} = \bar{X}(\mathbf{P}^*) = \frac{1}{M} \sum_{m=1}^{M} \hat{X}^{(m)}(p_m^*)$$

---

### 3. 수학적 수식 비교 요약

| 용어 (그래프 범례) | 의미 | 성분별 수식 ($t$번째 시간, $k$번째 채널) | 행렬 표기 |
| --- | --- | --- | --- |
| **`Target Y`** | 기준 신호 | $Y_{t, k}$ | $Y$ |
| **`Raw Mean X`** | 정렬 전 단순 평균 | $\displaystyle \bar{X}_{t, k}^{\text{raw}} = \frac{1}{M} \sum_{m=1}^{M} X_{t, k}^{(m)}$ | $\displaystyle \frac{1}{M} \sum_{m=1}^{M} X^{(m)}$ |
| **`Aligned Mean X`** | 정렬 후 최적 집계 평균 | $\displaystyle \bar{X}_{t, k}^{\text{aligned}} = \frac{1}{M} \sum_{m=1}^{M} X_{t, q_{m, k}^*}^{(m)}$ | $\displaystyle \frac{1}{M} \sum_{m=1}^{M} \hat{X}^{(m)}(p_m^*)$ |