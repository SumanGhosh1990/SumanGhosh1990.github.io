
ML Series - XGBoost , Use of scale_pos_weight in reducing class imbalance



---
🎉Ever wondered how Xgboost parameter scale_pos_weight can help in reducing imbalance of the data. What exactly is going under the hood, is it some kind of magic? Lets explore this together

---
In **XGBoost**, the parameter **`scale_pos_weight`** is specifically designed to help when you have **imbalanced classes** 

---

### 🔹 What it does

* **`scale_pos_weight`** controls the **relative weight** of positive examples (label = 1) compared to negative ones (label = 0).
* By default, all samples contribute equally to the loss. But when positives are rare, the model tends to ignore them and predict mostly negatives.
* Increasing `scale_pos_weight` tells the model: *"Pay more attention to misclassifying positives."*

---

### 🔹 How it affects training

XGBoost minimizes a loss function (e.g., logistic loss). Normally, each sample contributes equally:

$$
L = \sum_i w_i \cdot \ell(y_i, \hat{y}_i)
$$

When you set `scale_pos_weight = k`, every **positive sample**'s loss contribution is multiplied by **k**.

* Larger `scale_pos_weight` → positives penalized more if misclassified.
* This **shifts the decision boundary**, helping the model pick up minority cases.

---

### 🔹 How to choose the value

* 1000 negatives, 100 positives → `scale_pos_weight = 1000/100 = 10`.

But you should **tune it with cross-validation**, since the "right" value depends on the dataset and desired trade-off (precision vs recall) or any custom metric of interest say KS.

---

✅ In Essence:
`scale_pos_weight` in XGBoost **up-weights the positive class** so that the model does not ignore it, helping handle imbalanced data **without resampling**.

---
Let’s get straight into the **math behind `scale_pos_weight`** in XGBoost so you see exactly how it modifies the gradient updates.

---

### 1. XGBoost loss for binary classification

XGBoost often uses **logistic loss** (binary cross-entropy):

$$
\ell(y, \hat{p}) = - \big[ y \cdot \log(\hat{p}) + (1-y) \cdot \log(1-\hat{p}) \big]
$$

where

* $y \in \{0,1\}$ is the true label
* $\hat{p} = \sigma(\hat{y}) = \frac{1}{1+e^{-\hat{y}}}$ is the predicted probability
* $\hat{y}$ is the raw score (logit) from the tree.

---

### 2. Gradients and Hessians in XGBoost

XGBoost trains trees using **second-order Taylor expansion** of the loss.
For each sample $i$:

$$
g_i = \frac{\partial \ell}{\partial \hat{y}_i} = \hat{p}_i - y_i
$$

$$
h_i = \frac{\partial^2 \ell}{\partial \hat{y}_i^2} = \hat{p}_i(1 - \hat{p}_i)
$$

* $g_i$ is the **gradient** (first derivative).
* $h_i$ is the **Hessian** (second derivative).
* These control how much each sample pushes the tree split decisions.

---

### 3. Where `scale_pos_weight` comes in

When you set `scale_pos_weight = w`, XGBoost **multiplies both gradient and Hessian for positive samples** by $w$:

$$
g_i =
\begin{cases} 
(\hat{p}_i - y_i) \cdot w, & \text{if } y_i = 1 \\
(\hat{p}_i - y_i), & \text{if } y_i = 0
\end{cases}
$$

$$
h_i =
\begin{cases} 
\hat{p}_i (1-\hat{p}_i) \cdot w, & \text{if } y_i = 1 \\
\hat{p}_i (1-\hat{p}_i), & \text{if } y_i = 0
\end{cases}
$$

---

### 4. Effect on tree construction

When XGBoost builds a split, it calculates **gain**:

$$
\text{Gain} = \frac{1}{2} \left( \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right) - \gamma
$$

where

* $G = \sum g_i$ (sum of gradients in a node)
* $H = \sum h_i$ (sum of Hessians in a node)

👉 Because positive samples have **larger $g_i$ and $h_i$** after weighting, they **pull the tree splits more strongly toward classifying positives correctly**.

---

### 5. Intuition

* Without weighting: rare positives hardly move the gradients → model mostly predicts 0.
* With weighting: each positive contributes $w$ times more → the model **pays extra attention** to fixing their mistakes.
* This directly alters the optimization trajectory at each boosting round, not just the final threshold.

---

✅ So mathematically, `scale_pos_weight` is not just a class weight.
It **scales the derivatives** that drive tree growth → positives shape splits more strongly → more balanced decision boundary.

---
let’s build a **toy example** to see how `scale_pos_weight` changes the math inside XGBoost.

---

## ⚡ Setup

* **Dataset**: 3 negatives (y=0), 1 positive (y=1).
* Predicted probabilities ($\hat{p}$) before split:

| Sample | y (true) | $\hat{p}$ (pred) |
| ------ | -------- | ---------------- |
| A      | 0        | 0.2              |
| B      | 0        | 0.1              |
| C      | 0        | 0.3              |
| D      | 1        | 0.4              |

* Positive class (D) is minority (only 25%).
* Suppose we try a split that isolates D.

---

## 🔹 Step 1. Compute gradients & Hessians (no weight)

For logistic loss:

$$
g_i = \hat{p}_i - y_i, \quad h_i = \hat{p}_i (1-\hat{p}_i)
$$

* Sample A (y=0, p=0.2):
  $g = 0.2 - 0 = 0.2$, $h = 0.2(0.8)=0.16$
* Sample B (y=0, p=0.1):
  $g = 0.1 - 0 = 0.1$, $h = 0.1(0.9)=0.09$
* Sample C (y=0, p=0.3):
  $g = 0.3 - 0 = 0.3$, $h = 0.3(0.7)=0.21$
* Sample D (y=1, p=0.4):
  $g = 0.4 - 1 = -0.6$, $h = 0.4(0.6)=0.24$

**Totals**:

* Left node (A,B,C): $G_L = 0.2+0.1+0.3 = 0.6$, $H_L = 0.46$
* Right node (D): $G_R = -0.6$, $H_R = 0.24$

---

## 🔹 Step 2. Compute gain (no weight)

Split gain formula:

$$
\text{Gain} = \frac{1}{2} \left( \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right) - \gamma
$$

Let’s ignore regularization ($\lambda=0, \gamma=0$) for simplicity:

$$
\text{Gain} = \tfrac{1}{2} \left( \frac{0.6^2}{0.46} + \frac{(-0.6)^2}{0.24} - \frac{0^2}{0.7} \right)
$$

$$
= \tfrac{1}{2} \left( \frac{0.36}{0.46} + \frac{0.36}{0.24} \right)
= \tfrac{1}{2} (0.78 + 1.5) = 1.14
$$

So the split has **gain = 1.14**.

---

## 🔹 Step 3. Apply `scale_pos_weight = 3` (negatives/positives = 3/1)

Now, multiply gradients and Hessians for **positive sample (D)** by 3:

* Sample D (after weight):
  $g = -0.6 \times 3 = -1.8$, $h = 0.24 \times 3 = 0.72$

**New totals**:

* Left node (A,B,C): same as before $G_L = 0.6, H_L=0.46$
* Right node (D): $G_R = -1.8, H_R=0.72$

---

## 🔹 Step 4. Compute gain (with weight)

$$
\text{Gain} = \tfrac{1}{2} \left( \frac{0.6^2}{0.46} + \frac{(-1.8)^2}{0.72} - \frac{(0.6-1.8)^2}{0.46+0.72} \right)
$$

$$
= \tfrac{1}{2} \left( \frac{0.36}{0.46} + \frac{3.24}{0.72} - \frac{(-1.2)^2}{1.18} \right)
$$

$$
= \tfrac{1}{2} \left( 0.78 + 4.5 - 1.22 \right) 
= \tfrac{1}{2} (4.06) = 2.03
$$

Now the split gain is **2.03 (almost double!)**.

---

## ✅ Takeaway

* Without weighting, the split gain = **1.14** (not very strong).
* With `scale_pos_weight=3`, gain = **2.03** → the split isolating the minority positive class looks **much more attractive** to the tree.
* This is how `scale_pos_weight` **directly amplifies the influence of rare positives in gradient/Hessian calculations** → making splits more favorable toward catching minority cases.



