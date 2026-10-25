# Note tecniche — Hybrid Balance Control for WBR via Residual RL Optimization

Zhang, Lu*, Cao, Fang — Nankai University. ACIRS 2025, pp. 59–66. DOI 10.1109/ACIRS66343.2025.11360885.
PDF: `mpx/paper/Hybrid_Balance_Control_for_Wheeled_Bipedal_Robot_via_Residual_Reinforcement_Learning_Optimization.pdf`

## Robot e cinematica
- 8 motori: hip lateral (abduzione) ×2, hip forward ×2, knee ×2, ruote ×2.
- **Abduzione esclusa** (didascalia Fig. 2): *"The hip lateral motors allow for the abduction and adduction of the legs… In this work, the rotation of the hip lateral motors is not considered."* Non dicono come la tengono ferma.
- Frame {C}: origine nel punto medio dell'asse delle ruote, x in avanti. Frame {B}: IMU sulla base.
- q = [θ1l, θ1r, θ2l, θ2r, θ3l, θ3r, α, β, γ], cioè hip forward, knee, ruota, roll, pitch, yaw.
- CoG: p_c(q) = Σ m_i ·ᶜp_i(q) / Σ m_i (eq. 1); Jacobiano J ∈ ℝ^{3×9} (eq. 2).

## Controllore base (BMC), τ_ζ = [τ_P, τ_B] ∈ ℝ⁶
- **Equilibrio (ruote):** pendolo inverso del secondo ordine (eq. 3), LQR u = −K(q_b − q_bd) (eq. 4).
  - q_b = [x_w, ẋ_w, θ_p, θ̇_p], con K* = [14, 10, 90, 10].
  - Lo yaw γ è integrato nell'LQR.
- **Postura (hip forward e knee):** PD su q_θd, ottenuto integrando q̇_θd da IK (eq. 5–6).
  - Obiettivi: altezza del CoG, roll e pitch, anti-slittamento laterale (ψ, χ).
  - Guadagni: hip kp = 85, kd = 12; knee kp = 75, kd = 15.
- Il BMC è volutamente semplice.

## RL: HRC
- **Azione:** τ_π ∈ ℝ⁶ coppie dirette su tutti e 6 i motori. **τ_C = τ_ζ + τ_π.**
  - Non sono dati né scala, né Nm massimi, né se l'uscita passa per una tanh.
- **Stato** s_t = [s_r, s_g, s_e, s_τ]:
  - s_r = [q_b, p_c, ṗ_c, q_θ, q̇_θ];
  - s_g = [q_bd, z_cd, ż_cd, q_ψd], **senza θ_pd** (rimosso di proposito, per la robustezza a terreni sconosciuti);
  - s_e = [e_t, e_{t−1}], errori calcolati da s_g;
  - s_τ = [τ_ζ^t, τ_ζ^{t−1}, τ_π^{t−1}, τ_C^{t−1}].
  - **Incongruenza:** la rete dichiara un input di 30 dimensioni, ma i componenti elencati sono più di 30.
- **Reward:** r = r_θlim + r_bp, con r_bp = r_b + r_z + r_ψ.
  - r_θlim penalizza il superamento dei limiti di hip e knee (solo il massimo).
  - r_b: errori di q_b senza θ_pd; si limita solo θ̇_p.
  - r_z: altezza del CoG z_c. r_ψ: postura q_ψ.
  - Ogni errore passa per f(x) = 1/(exp(x)+exp(−x)+2) − 2 come stampato. Interpretazione: σ(x)σ(−x) − 0.25, che ha massimo 0 a errore nullo.
  - Pesi non dati.
- **Domain randomization:** masse dei moduli, rumore su encoder e IMU, attrito, angolo della pendenza.
- **Rete:** 2 layer FC da 1024 e 512, tanh, input 30, output 6.
- **SAC:** episodio massimo 5000 step, update ogni 1000 step, batch 512, lr 1e-3, γ 0.99, α iniziale 0.1, reward scale 10.
- **Hardware:** CPU 12 core, RTX 3060.
- **Algoritmo 1:** per ogni epoca ed episodio, τ_C = τ_ζ + τ_π, i dati vanno nel replay buffer, poi update SAC ed eval per epoca.

## Training
- **Group 1:** altezza costante; target di posizione casuale da 1 a 4 m; pendenza 0, 5 o 10°; q_ψd = 0.
- **Group 2:** piano; target da 1 a 4 m; cambio di altezza del CoG da 0.05 a 0.15 m. **Escluso da noi**, perché teniamo il CoM a 0.40.
- 3 seed, 40 epoche, 2M campioni. HRC converge in circa 20 epoche. SAC e DDPG puri falliscono; BMC-DDPG converge prematuramente.

## Esperimenti e numeri originali (target)
- **Task 1:** 2.5 m con altezza costante. **Task 2:** 2.5 m con +0.12 m di altezza (escluso).
- Sul piano, roll e yaw restano sotto 0.001 rad.

### Tab. I — RMSE
| Task | Slope | Metodo | x_w (m) | v (m/s) | θ_p (rad) | z_c (m) | β (rad) |
|---|---|---|---|---|---|---|---|
| 1 | 0° | BMC | 0.2475 | 0.0533 | 0.0157 | 0.0168 | 0.0065 |
| 1 | 0° | BMC-DDPG | 0.0966 | 0.0265 | 0.0128 | 0.0222 | 0.0039 |
| 1 | 0° | HRC | 0.0134 | 0.0119 | 0.0106 | 0.0032 | 0.0057 |
| 1 | 5° | BMC | 0.1397 | 0.0400 | 0.0177 | 0.0167 | 0.0064 |
| 1 | 5° | BMC-DDPG | 0.1032 | 0.0127 | 0.0134 | 0.0051 | 0.0049 |
| 1 | 5° | HRC | 0.0401 | 0.0099 | 0.0114 | 0.0015 | 0.0042 |
| 1 | 10° | BMC | 0.3436 | 0.0619 | 0.0461 | 0.0170 | 0.0106 |
| 1 | 10° | BMC-DDPG | 0.1193 | 0.0125 | 0.0313 | 0.0023 | 0.0128 |
| 1 | 10° | HRC | 0.0366 | 0.0086 | 0.0289 | 0.0012 | 0.0103 |
| 2 | 0° | BMC | 0.1909 | 0.0522 | 0.0123 | 0.0161 | 0.0069 |
| 2 | 0° | BMC-DDPG | 0.0548 | 0.0225 | 0.0084 | 0.0175 | 0.0058 |
| 2 | 0° | HRC | 0.0578 | 0.0176 | 0.0075 | 0.0018 | 0.0028 |

- **Errore di x_w a regime, piano:** BMC 0.235 m, HRC 0.013 m (−94%). z_c: da 0.017 a 0.0028 m.
- **Errore di x_w a regime a 10°:** da 0.39 a 0.03 m (−89% di RMSE).

### Spinte (Fig. 7)
- Task 1 sul piano, forze istantanee da 10, 20 e 30 N a 12, 20 e 32 s, in direzione opposta.
- Con 20 N, HRC si sposta di 0.2 m e torna in 4 s, mentre BMC si sposta di 0.5 m e non si stabilizza entro 12 s.

### Traiettorie (Fig. 8, 9)
- Traiettoria polinomiale di 5° ordine tra punti casuali e traiettoria multi-target ad alta dinamica.
- Multi-altezza 0.20 / 0.25 / 0.21 m: esclusa.

### Tab. II — LQR alternativi, Task 1
| K | Slope | Metrica | BMC | BMC-DDPG | HRC |
|---|---|---|---|---|---|
| K1 = [24, 8, 50, 9] | 0° | x_w | 0.0976 | 0.0557 | 0.0286 |
| | 0° | θ_p | 0.0286 | 0.0141 | 0.0101 |
| | 10° | x_w | 0.0820 | 0.1032 | 0.0237 |
| | 10° | θ_p | 0.0356 | 0.0155 | 0.0192 |
| K2 = [35, 15, 100, 15] | 0° | x_w | 0.1251 | 0.1272 | 0.0290 |
| | 0° | θ_p | 0.0164 | 0.0114 | 0.0105 |
| | 10° | x_w | 0.2056 | 0.0343 | 0.0201 |
| | 10° | θ_p | 0.0453 | 0.0166 | 0.0174 |

Simulatore: Webots. Solo simulazione, nessun test sul robot reale.
