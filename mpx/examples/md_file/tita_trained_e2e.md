# TITA E2E — Recap dei risultati sperimentali

Documento di riepilogo dei risultati ottenuti variando **leg `Kp`** e **wheel `action_scale_vel`**.
Da leggere insieme a `TITA_E2E_analisi_e_piano.md` (analisi teorica e formule).

---

## 1. Tabella riassuntiva degli esperimenti

| # | `Kp` | `action_scale_vel` | reward eval | varianza (±) | comportamento | valido? |
|---|---|---|---|---|---|---|
| A | 35 | 5  | ≈ 10.3   | n.d.  | bilanciamento ok; **tracking lineare scarso**; sotto‑traccia i comandi alti | ✗ (sotto‑tracking) |
| B | 80 | 5  | 20.994   | ± 1.561 | reward alto, tracking migliore, **ma salta/hoppa** per avanzare | ✗ (exploit) |
| C | 35 | 12 | 10.611   | ± 4.401 | più autorità ruote; migliora poco e resta **chiaramente peggiore** | ~ (parziale) |
| D | 50 | 12 | 20.841   | ± 4.467 | ottimo tracking, **niente salti**, locomozione su ruote | ✅ (buono) |
| E | 50 | 25 | ≈ 22.0   | **± 0.5** | **miglior tracking, varianza molto più bassa** | ✅ **migliore** |

Residuo aperto: un po' di **apertura/spreading delle gambe** (in analisi separata).

---

## 2. Lettura fattoriale (Kp × action_scale_vel)

```
                 asv = 5                    asv = 12                 asv = 25
   Kp = 35   A: sotto-tracking (soffitto)  C: 10.6 ± 4.4, parziale   —
   Kp = 50   —                             D: 20.8 ± 4.5, buono      E: 22.0 ± 0.5, MIGLIORE
   Kp = 80   B: 21.0 ± 1.6, hopping        —                         —
```

**Conclusioni empiriche:**

1. **`action_scale_vel` è necessario ma non sufficiente.**
   A→C (stesso `Kp=35`, `5→12`): reward da ~10 a ~10.6 con **varianza altissima** (± 4.4). Alzare il soffitto da solo non basta con gambe deboli.
2. **`Kp` intermedio è lo sweet spot.**
   C→D (`Kp 35→50`, asv=12): reward da ~10.6 a ~20.8. Con `Kp=80` (B) l'autorità c'è ma viene spesa **saltando**.
3. **Più autorità di velocità = molta meno varianza.**
   D→E (`asv 12→25`, `Kp=50`): reward ~20.8 → ~22, ma soprattutto la **varianza crolla da ± 4.5 a ± 0.5** (~9×). Il tracking diventa **consistente e robusto**.
4. Il miglior risultato nasce dall'**interazione** tra sufficiente autorità di velocità delle ruote e sufficiente rigidezza delle gambe — nessuno dei due parametri da solo basta.

---

## 3. Perché `asv=25` riduce così tanto la varianza

Velocità massima di rotolamento `v_max = asv * r` (r = 0.0925 m) e azione richiesta per `vx=1.0`:

| asv | v_max (a=1) | azione per vx=1.0 | margine |
|---|---|---|---|
| 12 | 1.11 m/s | a ≈ **0.90** (quasi saturo) | scarso |
| 25 | 2.31 m/s | a ≈ **0.43** (regione comoda) | ampio |

Con `asv=12` per i comandi alti la policy lavora **al limite della saturazione** dell'azione (`a→1`): poco margine sia per la propulsione sia per le correzioni di equilibrio → alcuni episodi/comandi "faticano" → **varianza alta**. Con `asv=25` lo stesso `vx=1.0` richiede solo `a≈0.43`: la policy opera nella **regione lineare/comoda** dello spazio d'azione, con ampio margine per propulsione **e** bilanciamento contemporaneamente → **tracking consistente, varianza bassa**.

**Nota (da tenere d'occhio):** il feedforward ruota è `Kd_wheel * asv` per unità d'azione; con `Kd_wheel=5.0` e `asv=25` diventa molto alto (transitori vicino al limite di coppia ±120 N·m e ruote più "twitchy"). Ha funzionato bene qui perché la policy opera a `a≈0.43`, ma **non conviene salire molto oltre `asv=25`**: si perde risoluzione a bassa velocità e si rischia twitchiness. Se emergessero transitori bruschi, la modifica isolata è ridurre `Kd_wheel` (es. 5.0→2.0) mantenendo `asv=25`.

---

## 4. Coerenza con l'analisi teorica

I risultati **confermano** le previsioni del report:

- **Il soffitto di rotolamento era il vincolo dominante.** A `asv=5` → `v_max ≈ 0.46 m/s`: comando `~0.94` irraggiungibile in rotolamento (A sotto‑traccia; B salta per aggirarlo). Alzando `asv` il comando diventa raggiungibile (C, D, E).
- **`Kp` intermedio = sweet spot.** Feedforward gamba `= Kp * action_scale_pos (0.5)`:
  - `Kp=35 → 17.5 N·m`: marginale nel transitorio di accelerazione → tracking mediocre e varianza alta anche con ruote potenti (C).
  - `Kp=50 → 25 N·m`: adeguato, non esplosivo → rotolamento pulito (D, E).
  - `Kp=80 → 40 N·m`: abbastanza da lanciare la base → salto (B).
- **Margine di autorità = robustezza.** Il crollo di varianza D→E è la conferma diretta che operare lontano dalla saturazione dell'azione rende il tracking robusto.

Configurazione **E** = il TEST 1 del piano (`Kp→50`, alzare `action_scale_vel`) portato un passo oltre sull'autorità ruote, ora validato.

---

## 5. Configurazione migliore attuale

```text
Kp               = 50
action_scale_vel = 25
# reward eval ≈ 22.0 ± 0.5
# (esperimenti sopra con a=[1.0,0.5]; prossimo run pulito: a=[1.0,0.0])
```

Comportamento: tracking del comando molto buono e **consistente** (varianza bassa), `tracking_lin_vel` migliore, locomozione **guidata dalle ruote**, **nessun hopping**.

---

## 6. Problema residuo: apertura delle gambe (leg spreading)

**Ipotesi principale (da verificare):** il gate del `posture` si spegne alle alte velocità
`gate = exp(-vx^2/0.25)` → a `vx=1.0` vale `~0.018`. Con il richiamo alla postura nominale quasi assente, i giunti di **abduzione/anca** sono liberi di derivare verso l'apertura durante la locomozione veloce.

Da controllare nei plot prima di intervenire:
- quali giunti si aprono (abduzione anca vs altro) e a quale `vx` inizia;
- se l'apertura correla con `gate → 0` (compare solo ad alta velocità);
- se è simmetrica (deriva posturale) o asimmetrica (compensazione di stabilità laterale).

Modifiche minime possibili (una alla volta, **solo se** confermato):
- **floor sul gate del posture:** `gate = clip(exp(-vx^2/0.25), gate_min, 1)` con `gate_min ≈ 0.1–0.2`;
- oppure una **regolarizzazione leggera e indipendente dal comando** solo sui giunti di abduzione (peso piccolo).

Da non fare: rimettere il `posture` pieno ad alta velocità, né cambiare più termini insieme.

---

## 7. Prossimi passi aggiornati

1. **Consolidare E su `a=[1.0,0.0]`** (task lineare puro, rumore/rand off) con `Kp=50`, `action_scale_vel=25`, `tracking_lin_vel` moderato, `tracking_ang_vel → 0.1`. Verificare che tracking e bassa varianza restino, senza hopping.
2. **Leg spreading:** diagnosticare come sopra; se confermato il meccanismo del gate, provare **solo** il `gate_min` (modifica isolata).
3. **Solo se servono transitori più puliti:** ridurre `Kd_wheel` (5.0→2.0) mantenendo `asv=25`.
4. **Solo dopo**, se resta sotto‑tracking agli estremi: PPO `discounting 0.99→0.997`, poi `entropy_cost 1e-2→2e-2` (una alla volta).

Ordine di priorità:

```
TRY FIRST      consolidare Kp=50 / action_scale_vel=25 su a=[1.0,0.0]
TRY SECOND     leg spreading -> gate_min ≈ 0.1-0.2 sul posture (se confermato)
TRY THIRD      Kd_wheel 5.0->2.0 (se transitori ruota bruschi)
ONLY IF NEEDED reward_vz = -2.0*vz^2 (se ricompare hopping) ; poi PPO discounting/entropy
```

---

## Appendice — mapping parametri → grandezze fisiche

```
v_max_roll (a=1) = action_scale_vel * r     ; r = 0.0925 m
  asv = 5  -> ~0.46 m/s   (soffitto -> A sotto-traccia, B salta)
  asv = 12 -> ~1.11 m/s   (raggiungibile ma a~0.90 -> varianza alta: C,D)
  asv = 25 -> ~2.31 m/s   (a~0.43 per vx=1.0 -> margine ampio, varianza bassa: E)

leg feedforward (a=1) = Kp * action_scale_pos ; action_scale_pos = 0.5
  Kp = 35 -> 17.5 N*m   (marginale in transitorio -> C)
  Kp = 50 -> 25.0 N*m   (adeguato, non esplosivo -> D,E)
  Kp = 80 -> 40.0 N*m   (esplosivo -> hopping B)

wheel feedforward gain = Kd_wheel * action_scale_vel
  Kd_wheel=5.0, asv=25 -> 125 N*m/azione (alto: sorvegliare saturazione ±120 e twitchiness)

posture gate = exp(-vx^2/0.25)  ; vx=1.0 -> 0.018  (sospetto leg spreading)
```