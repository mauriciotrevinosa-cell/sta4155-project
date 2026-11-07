"""
STA 4155 Group Project - complete analysis (Python).

Study 1: simple linear regression   (Fires.csv,    Chapters 4, 16, 17)
Study 2: multiple linear regression (Salaries.csv, Chapters 18, 19)

Run from anywhere:   python "claude version/analysis.py"

Everything in the report is computed here from the two CSV files:
  output/figures/*.png   graphs
  output/tables/*.csv    tables
  output/tables/*.txt    raw statsmodels output
  output/report.html     the written report (letter size, double spaced, 11.5 pt)
  output/report.pdf      same report printed to PDF (if Edge/Chrome is installed)
"""

from pathlib import Path
import itertools
import shutil
import subprocess
import sys
import tempfile
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # folder with Fires.csv / Salaries.csv
LOCAL_PKGS = ROOT / ".packages"          # optional vendored libraries
if LOCAL_PKGS.exists():
    sys.path.insert(0, str(LOCAL_PKGS))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy import stats

OUT = HERE / "output"
FIG = OUT / "figures"
TAB = OUT / "tables"
for d in (OUT, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
BLUE, ORANGE, GRAY, RED = "#1f5f99", "#d9822b", "#7a7a7a", "#b33a3a"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def fmt_p(p, html=True):
    """APA-style p-value text ('<' is escaped so it does not break the HTML report)."""
    if p < 0.001:
        return "p &lt; 0.001" if html else "p < 0.001"
    return f"p = {p:.3f}"


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


def save_text(name, text):
    (TAB / name).write_text(text, encoding="utf-8")
    return text


def html_table(df, digits=4, index=True):
    return df.to_html(float_format=lambda v: f"{v:.{digits}f}", index=index,
                      border=0, classes="tbl", na_rep="")


def coef_table(model, robust=None):
    """Coefficient table like R's summary(lm), optionally with HC3 p-values."""
    ci = model.conf_int(ALPHA)
    t = pd.DataFrame({"Estimate": model.params, "Std. Error": model.bse,
                      "t value": model.tvalues, "Pr(>|t|)": model.pvalues,
                      "95% CI low": ci[0], "95% CI high": ci[1]})
    if robust is not None:
        t["HC3 p-value"] = robust.pvalues
    return t


def anova_slr(model):
    """Classical ANOVA table (Regression / Residual / Total)."""
    ssr, sse = model.ess, model.ssr
    dfr, dfe = model.df_model, model.df_resid
    return pd.DataFrame(
        {"df": [dfr, dfe, dfr + dfe], "Sum Sq": [ssr, sse, ssr + sse],
         "Mean Sq": [ssr / dfr, sse / dfe, np.nan],
         "F value": [model.fvalue, np.nan, np.nan],
         "Pr(>F)": [model.f_pvalue, np.nan, np.nan]},
        index=["Regression", "Residual", "Total"])


def diagnostics(model):
    infl = model.get_influence()
    return pd.DataFrame({
        "fitted": model.fittedvalues, "residual": model.resid,
        "stud_resid": infl.resid_studentized_external,
        "leverage": infl.hat_matrix_diag, "cooks_d": infl.cooks_distance[0]})


def pre(text):
    return f"<pre>{text}</pre>"


report = []            # HTML fragments, joined at the end
log = []               # console summary


def say(line):
    print(line)
    log.append(line)


# ===========================================================================
# STUDY 1 - FIRE DAMAGE vs DISTANCE (simple linear regression)
# ===========================================================================
fires = pd.read_csv(ROOT / "Fires.csv")
fires.index = range(1, len(fires) + 1)          # fire number 1..15
assert not fires.isna().any().any(), "Fires.csv has missing values"
n1 = len(fires)

# --- 1.1 Exploratory: descriptive statistics and correlation (Ch. 4)
desc1 = fires.describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
desc1.to_csv(TAB / "fires_descriptive.csv")
r, r_p = stats.pearsonr(fires.DISTANCE, fires.DAMAGE)

# --- 1.2 Least-squares fit (Ch. 4, 16)
fit1 = smf.ols("DAMAGE ~ DISTANCE", data=fires).fit()
b0, b1 = fit1.params
ci1 = fit1.conf_int(ALPHA)
coef1 = coef_table(fit1)
coef1.to_csv(TAB / "fires_coefficients.csv")
aov1 = anova_slr(fit1)
aov1.to_csv(TAB / "fires_anova.csv")
save_text("fires_summary.txt", fit1.summary().as_text())
s_e = np.sqrt(fit1.mse_resid)

# --- 1.3 Assumption checks (Ch. 16, 17)
d1 = diagnostics(fit1)
d1.to_csv(TAB / "fires_diagnostics.csv")
sw1 = stats.shapiro(fit1.resid)
bp1 = het_breuschpagan(fit1.resid, fit1.model.exog)
dw1 = durbin_watson(fit1.resid)
DW_L, DW_U = 1.08, 1.36           # Durbin-Watson table, n = 15, k = 1, alpha = .05
lev_cut1 = 2 * 2 / n1             # 2(k+1)/n
hi_lev1 = d1.index[d1.leverage > lev_cut1].tolist()
big_res1 = d1.index[d1.stud_resid.abs() > 2].tolist()
max_cook1 = d1.cooks_d.idxmax()

# Sensitivity: refit without the most influential fire
fit1_drop = smf.ols("DAMAGE ~ DISTANCE", data=fires.drop(index=max_cook1)).fit()

# --- 1.4 Estimation and prediction (Ch. 16)
x_new = pd.DataFrame({"DISTANCE": [1.0, 2.0, 3.5, 5.0, 6.0]})
pr1 = fit1.get_prediction(x_new).summary_frame(alpha=ALPHA)
pred1 = pd.concat([x_new, pr1[["mean", "mean_ci_lower", "mean_ci_upper",
                               "obs_ci_lower", "obs_ci_upper"]]], axis=1)
pred1.columns = ["Distance (mi)", "Predicted damage", "CI mean low", "CI mean high",
                 "PI low", "PI high"]
pred1.to_csv(TAB / "fires_predictions.csv", index=False)
row35 = pred1.loc[pred1["Distance (mi)"] == 3.5].iloc[0]

# --- 1.5 Figures
grid = pd.DataFrame({"DISTANCE": np.linspace(fires.DISTANCE.min(), fires.DISTANCE.max(), 200)})
band = fit1.get_prediction(grid).summary_frame(alpha=ALPHA)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.fill_between(grid.DISTANCE, band.obs_ci_lower, band.obs_ci_upper, color=GRAY, alpha=0.15,
                label="95% prediction band (one new fire)")
ax.fill_between(grid.DISTANCE, band.mean_ci_lower, band.mean_ci_upper, color=BLUE, alpha=0.25,
                label="95% confidence band (mean damage)")
ax.plot(grid.DISTANCE, band["mean"], color=BLUE, lw=2,
        label=f"LS line: ŷ = {b0:.3f} + {b1:.3f}x")
ax.scatter(fires.DISTANCE, fires.DAMAGE, color=ORANGE, edgecolor="k", zorder=3, label="Observed fires")
for i in [max_cook1]:
    ax.annotate(f"fire {i}", (fires.DISTANCE[i], fires.DAMAGE[i]), xytext=(-40, 5),
                textcoords="offset points", fontsize=8)
ax.set(xlabel="Distance to nearest fire station (miles)",
       ylabel="Fire damage (thousands of $)",
       title="Figure 1. Fire damage vs. distance with 95% bands")
ax.legend(fontsize=8, loc="upper left")
FIG1 = save_fig(fig, "fig1_fires_fit.png")

fig, axs = plt.subplots(2, 2, figsize=(8, 6))
ax = axs[0, 0]
ax.scatter(d1.fitted, d1.residual, color=BLUE)
ax.axhline(0, color=RED)
ax.set(xlabel="Fitted damage", ylabel="Residual", title="(a) Residuals vs fitted")
ax = axs[0, 1]
ax.plot(d1.index, d1.residual, marker="o", color=BLUE)
ax.axhline(0, color=RED)
ax.set(xlabel="Case index (order in file)", ylabel="Residual", title="(b) Residuals vs order")
ax = axs[1, 0]
sm.qqplot(fit1.resid, line="s", ax=ax, markerfacecolor=BLUE, markeredgecolor=BLUE)
ax.set_title("(c) Normal Q-Q plot of residuals")
ax = axs[1, 1]
ax.scatter(d1.leverage, d1.stud_resid, s=30 + 400 * d1.cooks_d, color=BLUE, alpha=0.7)
ax.axvline(lev_cut1, color=GRAY, ls="--", label="2(k+1)/n")
ax.axhline(2, color=RED, ls=":"); ax.axhline(-2, color=RED, ls=":")
for i in d1.index[(d1.leverage > lev_cut1) | (d1.stud_resid.abs() > 2)]:
    ax.annotate(str(i), (d1.leverage[i], d1.stud_resid[i]), fontsize=8, xytext=(3, 3),
                textcoords="offset points")
ax.set(xlabel="Leverage", ylabel="Studentized residual",
       title="(d) Influence (size ∝ Cook's D)")
ax.legend(fontsize=8)
fig.suptitle("Figure 2. Residual diagnostics, fire damage model", y=1.02)
FIG2 = save_fig(fig, "fig2_fires_diagnostics.png")

say(f"[Study 1] damage = {b0:.4f} + {b1:.4f} distance, R2 = {fit1.rsquared:.4f}, "
    f"slope {fmt_p(fit1.pvalues['DISTANCE'], html=False)}")


# ===========================================================================
# STUDY 2 - EXECUTIVE SALARIES (multiple linear regression)
# ===========================================================================
sal = pd.read_csv(ROOT / "Salaries.csv")
assert not sal.isna().any().any(), "Salaries.csv has missing values"
n2 = len(sal)
PRED = [f"X{i}" for i in range(1, 11)]
QUANT = ["X1", "X2", "X4", "X5", "X7", "X8", "X10"]
BINARY = ["X3", "X6", "X9"]
LABEL = {"X1": "Experience (yrs)", "X2": "Education (yrs)", "X3": "Bonus eligible",
         "X4": "Employees supervised", "X5": "Corporate assets ($M)", "X6": "Board member",
         "X7": "Age (yrs)", "X8": "Company profits ($M)", "X9": "International resp.",
         "X10": "Company sales ($M)"}

# --- 2.1 Exploratory analysis
desc2 = sal[["Y"] + PRED].describe().T[["mean", "std", "min", "50%", "max"]]
desc2.insert(0, "Variable", ["ln(Salary)"] + [LABEL[p] for p in PRED])
desc2.to_csv(TAB / "salaries_descriptive.csv")
corr = sal[["Y"] + PRED].corr()
corr.to_csv(TAB / "salaries_correlation.csv")

fig, ax = plt.subplots(figsize=(7, 5.8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(11), corr.columns); ax.set_yticks(range(11), corr.columns)
for i in range(11):
    for j in range(11):
        ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(corr.iat[i, j]) > 0.6 else "black")
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Figure 3. Correlation matrix, Salaries data")
FIG3 = save_fig(fig, "fig3_salaries_correlation.png")

fig, axs = plt.subplots(2, 5, figsize=(13, 5.5), sharey=True)
for ax, v in zip(axs.flat, PRED):
    if v in BINARY:
        groups = [sal.loc[sal[v] == g, "Y"] for g in (0, 1)]
        ax.boxplot(groups, tick_labels=["0 = no", "1 = yes"], widths=0.5)
    else:
        ax.scatter(sal[v], sal.Y, s=12, color=BLUE, alpha=0.7)
        z = np.polyfit(sal[v], sal.Y, 1)
        xs = np.linspace(sal[v].min(), sal[v].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color=RED, lw=1)
    ax.set_title(f"{v}: {LABEL[v]}", fontsize=9)
for ax in axs[:, 0]:
    ax.set_ylabel("Y = ln(Salary)")
fig.suptitle("Figure 4. ln(Salary) against each potential predictor", y=1.01)
FIG4 = save_fig(fig, "fig4_salaries_pairs.png")

# --- 2.2 Full first-order model with all 10 predictors (Ch. 18)
f_full = "Y ~ " + " + ".join(PRED)
full = smf.ols(f_full, data=sal).fit()
coef_full = coef_table(full)
coef_full.to_csv(TAB / "salaries_full_coefficients.csv")
save_text("salaries_full_summary.txt", full.summary().as_text())
Xf = full.model.exog
vif = pd.DataFrame({"Variable": [LABEL[p] for p in PRED],
                    "VIF": [variance_inflation_factor(Xf, i) for i in range(1, Xf.shape[1])]},
                   index=PRED)
vif.to_csv(TAB / "salaries_vif.csv")
nonsig_full = [p for p in PRED if full.pvalues[p] > ALPHA]
sig_full = [p for p in PRED if full.pvalues[p] <= ALPHA]

# --- 2.3 Variable selection (Ch. 19)
# (a) Partial F-test: can the non-significant predictors be dropped together?
f_red = "Y ~ " + " + ".join(sig_full)
red = smf.ols(f_red, data=sal).fit()
pf_drop = anova_lm(red, full)
save_text("salaries_partialF_drop.txt", pf_drop.to_string())
pF_drop, pP_drop = pf_drop["F"].iloc[1], pf_drop["Pr(>F)"].iloc[1]

# (b) All-subsets regression: best model of each size by adjusted R2, with Cp and BIC
mse_full = full.mse_resid
rows = []
for k in range(1, 11):
    for combo in itertools.combinations(PRED, k):
        m = smf.ols("Y ~ " + " + ".join(combo), data=sal).fit()
        cp = m.ssr / mse_full - (n2 - 2 * (k + 1))
        rows.append({"k": k, "Predictors": ", ".join(combo), "R2": m.rsquared,
                     "Adj R2": m.rsquared_adj, "Cp": cp, "BIC": m.bic})
subsets = pd.DataFrame(rows)
best = subsets.loc[subsets.groupby("k")["Adj R2"].idxmax()].set_index("k")
best.to_csv(TAB / "salaries_best_subsets.csv")
best_adj_k = int(best["Adj R2"].idxmax())
best_bic_k = int(best["BIC"].idxmin())
cp_ok = best.index[(best.Cp <= best.index + 1 + 1e-9)].tolist()   # Cp <= p
best_cp_k = int(min(cp_ok)) if cp_ok else int(best.Cp.idxmin())

fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
axs[0].plot(best.index, best["Adj R2"], marker="o", color=BLUE)
axs[0].set(xlabel="Number of predictors", ylabel="Adjusted R²", title="(a) Adjusted R²")
axs[1].plot(best.index, best.Cp, marker="o", color=BLUE)
axs[1].plot(best.index, best.index + 1, ls="--", color=GRAY, label="Cp = p")
axs[1].set(xlabel="Number of predictors", ylabel="Mallows' Cp", title="(b) Mallows' Cp",
           ylim=(0, 30)); axs[1].legend(fontsize=8)
axs[2].plot(best.index, best.BIC, marker="o", color=BLUE)
axs[2].set(xlabel="Number of predictors", ylabel="BIC", title="(c) BIC (lower is better)")
for ax in axs:
    ax.axvline(len(sig_full), color=ORANGE, ls=":", lw=1.5)
fig.suptitle("Figure 5. Best-subsets regression: best model of each size", y=1.03)
FIG5 = save_fig(fig, "fig5_salaries_best_subsets.png")

# --- 2.4 Model refinement: curvature and interaction (Ch. 19)
# Residuals of the reduced model vs each retained quantitative predictor
red_res = red.resid
fig, axs = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
for ax, v in zip(axs, [p for p in sig_full if p in QUANT]):
    ax.scatter(sal[v], red_res, s=12, color=BLUE, alpha=0.7)
    lw = sm.nonparametric.lowess(red_res, sal[v], frac=0.6)
    ax.plot(lw[:, 0], lw[:, 1], color=RED, lw=2, label="LOWESS smooth")
    ax.axhline(0, color=GRAY, lw=1)
    ax.set(xlabel=LABEL[v], title=v)
axs[0].set_ylabel("Residual (reduced model)")
axs[0].legend(fontsize=8)
fig.suptitle("Figure 6. Residuals of the 5-predictor model vs each quantitative predictor", y=1.04)
FIG6 = save_fig(fig, "fig6_salaries_curvature.png")

# (a) Add a quadratic term in experience
f_quad = f_red + " + I(X1**2)"
quad = smf.ols(f_quad, data=sal).fit()
pf_quad = anova_lm(red, quad)
pF_quad, pP_quad = pf_quad["F"].iloc[1], pf_quad["Pr(>F)"].iloc[1]

# (b) Does the effect of each quantitative predictor depend on bonus eligibility?
quant_in = [p for p in sig_full if p in QUANT]
inter_terms = [f"X3:{v}" for v in quant_in]
f_allint = f_quad + " + " + " + ".join(inter_terms)
allint = smf.ols(f_allint, data=sal).fit()
pf_int = anova_lm(quad, allint)
pF_int, pP_int = pf_int["F"].iloc[1], pf_int["Pr(>F)"].iloc[1]
keep_int = [t for t in inter_terms if allint.pvalues[t] <= ALPHA]
f_final = f_quad + "".join(f" + {t}" for t in keep_int)
final = smf.ols(f_final, data=sal).fit()
pf_rest = anova_lm(final, allint)          # are the other interactions needed?
pF_rest, pP_rest = pf_rest["F"].iloc[1], pf_rest["Pr(>F)"].iloc[1]
save_text("salaries_model_building_tests.txt",
          "Reduced vs full (drop non-significant predictors)\n" + pf_drop.to_string()
          + "\n\nReduced vs reduced + X1^2\n" + pf_quad.to_string()
          + "\n\nQuadratic vs quadratic + all X3 interactions\n" + pf_int.to_string()
          + "\n\nFinal vs all X3 interactions\n" + pf_rest.to_string())

# Interaction plot: X4 vs Y by bonus eligibility
fig, ax = plt.subplots(figsize=(6.2, 4))
for g, col, lab in [(0, ORANGE, "Not bonus eligible (X3 = 0)"), (1, BLUE, "Bonus eligible (X3 = 1)")]:
    sub = sal[sal.X3 == g]
    ax.scatter(sub.X4, sub.Y, color=col, s=16, alpha=0.7, label=lab)
    xs = np.linspace(sal.X4.min(), sal.X4.max(), 50)
    newd = pd.DataFrame({"X1": sal.X1.mean(), "X2": sal.X2.mean(), "X3": g, "X4": xs,
                         "X5": sal.X5.mean()})
    ax.plot(xs, final.predict(newd), color=col, lw=2)
ax.set(xlabel="Number of employees supervised (X4)", ylabel="ln(Salary)",
       title="Figure 7. Interaction: bonus eligibility × employees supervised")
ax.legend(fontsize=8)
ax.text(0.99, 0.02, "Lines: final model, other predictors at their means", transform=ax.transAxes,
        ha="right", fontsize=7, color=GRAY)
FIG7 = save_fig(fig, "fig7_salaries_interaction.png")

# --- 2.5 Final model: inference, interpretation and diagnostics
final_hc3 = final.get_robustcov_results(cov_type="HC3")
coef_final = coef_table(final, robust=final_hc3)
coef_final.to_csv(TAB / "salaries_final_coefficients.csv")
save_text("salaries_final_summary.txt", final.summary().as_text())
save_text("salaries_final_HC3_summary.txt", final_hc3.summary().as_text())
d2 = diagnostics(final)
d2.index = sal.ID
d2.to_csv(TAB / "salaries_final_diagnostics.csv")
sw2 = stats.shapiro(final.resid)
bp2 = het_breuschpagan(final.resid, final.model.exog)
bp2_x3 = het_breuschpagan(final.resid, sm.add_constant(sal[["X3"]]))
dw2 = durbin_watson(final.resid)
sd_by_bonus = final.resid.groupby(sal.X3).std()
lev_cut2 = 2 * (final.df_model + 1) / n2
hi_lev2 = d2.index[d2.leverage > lev_cut2].tolist()
big_res2 = d2.index[d2.stud_resid.abs() > 2.5].tolist()
max_cook2 = int(d2.cooks_d.idxmax())
hc3_all_sig = bool((final_hc3.pvalues[1:] < ALPHA).all())
fit2_drop = smf.ols(f_final, data=sal[sal.ID != max_cook2]).fit()
max_rel_change = (np.abs(fit2_drop.params - final.params) / np.abs(final.params)).max()

# Percent interpretation of the main effects (log response)
pct = lambda b: 100 * np.expm1(b)
g = final.params
ci_f = final.conf_int(ALPHA)
x1_turn = -g["X1"] / (2 * g["I(X1 ** 2)"])
x1_marg = {x: pct(g["X1"] + 2 * g["I(X1 ** 2)"] * x) for x in (5, 13, 20)}
int_name = keep_int[0] if keep_int else None
x4_eff = {0: pct(100 * g["X4"])}
if int_name:
    x4_eff[1] = pct(100 * (g["X4"] + g[int_name]))
    bonus_eff = {x: pct(g["X3"] + g[int_name] * x) for x in (100, int(round(sal.X4.mean())), 600)}

interp = pd.DataFrame({
    "Term": ["X2: +1 year of education", "X5: +$10M corporate assets",
             "X4: +100 employees supervised (not bonus eligible)"],
    "Coefficient": [g["X2"], g["X5"], g["X4"]],
    "Effect on median salary (%)": [pct(g["X2"]), pct(10 * g["X5"]), x4_eff[0]],
    "95% CI low (%)": [pct(ci_f.loc["X2", 0]), pct(10 * ci_f.loc["X5", 0]), pct(100 * ci_f.loc["X4", 0])],
    "95% CI high (%)": [pct(ci_f.loc["X2", 1]), pct(10 * ci_f.loc["X5", 1]), pct(100 * ci_f.loc["X4", 1])],
}).set_index("Term")
interp.to_csv(TAB / "salaries_final_interpretation.csv")

# Final-model diagnostics figure
fig, axs = plt.subplots(2, 3, figsize=(12, 7))
ax = axs[0, 0]
ax.scatter(d2.fitted, d2.residual, s=14, color=BLUE)
ax.axhline(0, color=RED); ax.set(xlabel="Fitted ln(Salary)", ylabel="Residual", title="(a) Residuals vs fitted")
ax = axs[0, 1]
sm.qqplot(final.resid, line="s", ax=ax, markersize=4, markerfacecolor=BLUE, markeredgecolor=BLUE)
ax.set_title("(b) Normal Q-Q plot")
ax = axs[0, 2]
ax.hist(final.resid, bins=15, color=BLUE, alpha=0.8, edgecolor="white")
ax.set(xlabel="Residual", ylabel="Count", title="(c) Histogram of residuals")
ax = axs[1, 0]
ax.boxplot([final.resid[sal.X3 == 0], final.resid[sal.X3 == 1]],
           tick_labels=["Not eligible", "Bonus eligible"], widths=0.5)
ax.axhline(0, color=GRAY, lw=1)
ax.set(ylabel="Residual", title="(d) Residual spread by bonus eligibility")
ax = axs[1, 1]
ax.plot(sal.ID, final.resid, marker=".", lw=0.6, color=BLUE)
ax.axhline(0, color=RED); ax.set(xlabel="Executive ID", ylabel="Residual", title="(e) Residuals vs order")
ax = axs[1, 2]
ax.scatter(d2.leverage, d2.stud_resid, s=10 + 600 * d2.cooks_d, color=BLUE, alpha=0.6)
ax.axvline(lev_cut2, color=GRAY, ls="--", label="2(k+1)/n")
ax.axhline(2.5, color=RED, ls=":"); ax.axhline(-2.5, color=RED, ls=":")
for i in set(hi_lev2) | set(big_res2) | {max_cook2}:
    ax.annotate(str(i), (d2.leverage[i], d2.stud_resid[i]), fontsize=7, xytext=(3, 3),
                textcoords="offset points")
ax.set(xlabel="Leverage", ylabel="Studentized residual", title="(f) Influence (size ∝ Cook's D)")
ax.legend(fontsize=8)
fig.suptitle("Figure 8. Residual diagnostics, final salary model", y=1.01)
FIG8 = save_fig(fig, "fig8_salaries_diagnostics.png")

# --- 2.6 Out-of-sample check: repeated 10-fold cross-validation
rng = np.random.default_rng(4155)
cv_models = {"Full (10 predictors)": f_full, "Reduced (5 predictors)": f_red,
             "Final (5 + X1² + interaction)": f_final}
cv = {k: [] for k in cv_models}
for rep in range(20):
    folds = np.array_split(rng.permutation(n2), 10)
    for test in folds:
        train = np.setdiff1d(np.arange(n2), test)
        for name, f in cv_models.items():
            m = smf.ols(f, data=sal.iloc[train]).fit()
            err = sal.Y.iloc[test] - m.predict(sal.iloc[test])
            cv[name].append(np.mean(err ** 2))
cv_tab = pd.DataFrame({
    "Predictors (p)": [int(smf.ols(f, data=sal).fit().df_model) for f in cv_models.values()],
    "Adj R2": [smf.ols(f, data=sal).fit().rsquared_adj for f in cv_models.values()],
    "AIC": [smf.ols(f, data=sal).fit().aic for f in cv_models.values()],
    "CV RMSE (log scale)": [np.sqrt(np.mean(v)) for v in cv.values()]}, index=list(cv_models))
cv_tab.to_csv(TAB / "salaries_model_comparison.csv")
cv_best = cv_tab["CV RMSE (log scale)"].idxmin()

# --- 2.7 Prediction for a typical executive
exec_new = pd.DataFrame({"X1": [10], "X2": [16], "X3": [1], "X4": [300], "X5": [175]})
pr2 = final.get_prediction(exec_new).summary_frame(alpha=ALPHA).iloc[0]
pred_salary = np.exp(pr2["mean"])
pi_salary = np.exp([pr2.obs_ci_lower, pr2.obs_ci_upper])
ci_salary = np.exp([pr2.mean_ci_lower, pr2.mean_ci_upper])

say(f"[Study 2] final: {f_final}  adjR2 = {final.rsquared_adj:.4f}, best CV = {cv_best}")


# ===========================================================================
# REPORT (HTML -> PDF)
# ===========================================================================
eq1 = f"Damage&#770; = {b0:.3f} + {b1:.3f} &times; Distance"
sig_list = ", ".join(sig_full)
drop_list = ", ".join(nonsig_full)
vif_max = vif.VIF.max()
vif_max_var = vif.VIF.idxmax()

eq2_terms = []
for name, val in final.params.items():
    if name == "Intercept":
        continue
    nice = name.replace("I(X1 ** 2)", "X1<sup>2</sup>").replace(":", "&middot;")
    eq2_terms.append(f"{'&minus;' if val < 0 else '+'} {abs(val):.5f}&thinsp;{nice}")
eq2 = f"ln(Salary)&#770; = {g['Intercept']:.4f} " + " ".join(eq2_terms)

css = """
@page { size: letter; margin: 1in; }
body { font-family: 'Times New Roman', Georgia, serif; font-size: 12pt; line-height: 2;
       max-width: 6.5in; margin: 0 auto; color: #111; background: #fff; }
h1 { font-size: 18pt; text-align: center; line-height: 1.3; margin-bottom: 0; }
h2 { font-size: 14pt; border-bottom: 1px solid #999; margin-top: 1.6em; page-break-after: avoid; }
h3 { font-size: 12pt; margin-bottom: 0; page-break-after: avoid; }
p { margin: 0 0 0.6em 0; text-align: justify; }
.center { text-align: center; }
.title-page { text-align: center; page-break-after: always; padding-top: 1.5in; }
.eq { text-align: center; font-style: italic; margin: 0.4em 0; }
table.tbl { border-collapse: collapse; font-size: 9.5pt; line-height: 1.25; margin: 0.6em auto;
            page-break-inside: avoid; }
table.tbl th, table.tbl td { padding: 2px 7px; border-bottom: 1px solid #ccc; text-align: right; }
table.tbl th { border-bottom: 1.5px solid #333; }
table.tbl tbody th { text-align: left; font-weight: normal; }
table.members td, table.members th { border: 1px solid #555; padding: 4px 18px; line-height: 1.4; }
table.members { border-collapse: collapse; margin: 1em auto; font-size: 11pt; }
.caption { font-size: 10pt; text-align: center; line-height: 1.3; margin-bottom: 1em; }
img { display: block; max-width: 100%; margin: 0.5em auto 0.2em auto; page-break-inside: avoid; }
pre { font-family: Consolas, 'Courier New', monospace; font-size: 7.6pt; line-height: 1.2;
      background: #f6f6f6; padding: 6px; white-space: pre; overflow-x: auto; page-break-inside: avoid; }
.box { border: 1px solid #999; padding: 4px 12px; margin: 0.8em 0; line-height: 1.6; }
.appendix { page-break-before: always; }
"""

R = report.append

# ---- Title page
R(f"""<div class="title-page">
<h1>Regression Analysis of Fire Damage<br>and Executive Salaries</h1>
<p class="center">STA 4155 &mdash; Group Project<br>Baruch College</p>
<p class="center" style="margin-top:0.6in"><b>Group members and contribution</b></p>
<table class="members">
<tr><th>Member</th><th>Assigned part</th><th>Contribution (%)</th></tr>
<tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr>
<tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr>
<tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr>
<tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr>
</table>
<p class="center" style="font-size:10pt">All numerical results were produced with Python
(pandas, statsmodels, SciPy, matplotlib). The code is submitted separately (analysis.py).</p>
</div>""")

# ---- Summary
R(f"""<h2>Executive Summary</h2>
<p><b>Study 1.</b> Fire damage increases strongly and linearly with the distance between the
house and the nearest fire station (r = {r:.3f}). Each additional mile is associated with an
estimated increase of <b>${b1*1000:,.0f}</b> in average damage (95% CI ${ci1.loc['DISTANCE',0]*1000:,.0f}
to ${ci1.loc['DISTANCE',1]*1000:,.0f}). Distance explains {100*fit1.rsquared:.1f}% of the variation
in damage, and the regression assumptions are reasonably satisfied.</p>
<p><b>Study 2.</b> Of the ten candidate predictors, only experience (X1), education (X2), bonus
eligibility (X3), employees supervised (X4) and corporate assets (X5) are related to log salary
once the others are controlled for; board membership, age, profits, international responsibility
and sales add nothing (partial F = {pF_drop:.2f}, {fmt_p(pP_drop)}). The final model adds a
diminishing-returns term for experience and a bonus &times; supervision interaction. It explains
{100*final.rsquared:.1f}% of the variation in log salary (adjusted R&sup2; = {final.rsquared_adj:.3f})
and has the smallest cross-validated prediction error of the models compared.</p>""")

# ---- Study 1
R(f"""<h2>1. Study 1: Fire Damage and Distance to the Fire Station</h2>
<h3>1.1 Objective and data</h3>
<p>An insurance company wants to know whether, and how strongly, the damage from major residential
fires is related to the distance from the burning house to the nearest fire station. The sample
contains n = {n1} recent fires; the response is <i>damage</i> (thousands of dollars) and the
explanatory variable is <i>distance</i> (miles). There are no missing values.</p>
{html_table(desc1, 2)}
<p class="caption">Table 1. Descriptive statistics, Fires data.</p>
<h3>1.2 Exploratory analysis</h3>
<p>The scatterplot (Figure 1) shows a strong, positive, linear association with no visible
curvature and no obvious outliers. The sample correlation is <b>r = {r:.4f}</b>
({fmt_p(r_p)}), so a straight-line model is appropriate. Distances range from
{fires.DISTANCE.min()} to {fires.DISTANCE.max()} miles; conclusions apply only within this range.</p>
<img src="{FIG1}" style="width:5.6in">
<p class="caption">Figure 1. Least-squares line with 95% confidence band for mean damage
(dark) and 95% prediction band for an individual fire (light).</p>
<h3>1.3 Least-squares regression model</h3>
<p>The model is Damage = &beta;<sub>0</sub> + &beta;<sub>1</sub>Distance + &epsilon;, with
&epsilon; ~ N(0, &sigma;&sup2;) independent. The fitted line is</p>
<p class="eq">{eq1}</p>
{html_table(coef1, 4)}
<p class="caption">Table 2. Coefficient estimates, Study 1.</p>
<p><b>Slope.</b> For each additional mile from the fire station, average fire damage increases by
an estimated {b1:.3f} thousand dollars (about ${b1*1000:,.0f}). <b>Intercept.</b> The estimate
{b0:.3f} would be the average damage for a fire at zero distance; because the closest fire in the
sample was {fires.DISTANCE.min()} miles away, it is a slight extrapolation and should be read as
the baseline damage that occurs regardless of distance. The residual standard error is
s<sub>e</sub> = {s_e:.3f} thousand dollars.</p>
<h3>1.4 Inference about the slope and model fit</h3>
<p>To test H<sub>0</sub>: &beta;<sub>1</sub> = 0 against H<sub>a</sub>: &beta;<sub>1</sub> &ne; 0,
t = {fit1.tvalues['DISTANCE']:.3f} with {int(fit1.df_resid)} df ({fmt_p(fit1.pvalues['DISTANCE'])}).
We reject H<sub>0</sub>: distance is a significant linear predictor of damage. The 95% confidence
interval for &beta;<sub>1</sub> is ({ci1.loc['DISTANCE',0]:.3f}, {ci1.loc['DISTANCE',1]:.3f}) thousand
dollars per mile; it excludes zero. The ANOVA F statistic ({fit1.fvalue:.2f}) equals t&sup2;, as
expected in simple regression. The coefficient of determination is
<b>R&sup2; = {fit1.rsquared:.4f}</b> (= r&sup2;): {100*fit1.rsquared:.1f}% of the variability in fire
damage is explained by its linear relationship with distance.</p>
{html_table(aov1, 4)}
<p class="caption">Table 3. Analysis of variance, Study 1.</p>
<h3>1.5 Checking the assumptions</h3>
<img src="{FIG2}" style="width:6.2in">
<p class="caption">Figure 2. Residual plots for the fire damage model.</p>
<p><b>Linearity and constant variance.</b> The residuals vs fitted values (Figure 2a) scatter
randomly around zero with no curve or funnel shape. The Breusch&ndash;Pagan test does not detect
non-constant variance (LM = {bp1[0]:.3f}, {fmt_p(bp1[1])}).
<b>Normality.</b> The points in the normal Q-Q plot (Figure 2c) lie close to the line, and the
Shapiro&ndash;Wilk test does not reject normality (W = {sw1.statistic:.4f}, {fmt_p(sw1.pvalue)}).
<b>Independence.</b> The fires were sampled from different houses, which supports independence.
As a check, the Durbin&ndash;Watson statistic is d = {dw1:.3f}; with n = 15 and one predictor the 5%
bounds are d<sub>L</sub> = {DW_L} and d<sub>U</sub> = {DW_U}, so the test is
{"inconclusive" if DW_L <= dw1 <= DW_U else ("not significant" if dw1 > DW_U else "significant")}.
Because the file order is not a time order, this is a weak concern; Figure 2b shows no
systematic pattern.</p>
<p><b>Unusual observations.</b> {"No studentized residual exceeds &plusmn;2, so there are no outliers." if not big_res1 else f"Fire(s) {', '.join(map(str, big_res1))} have studentized residuals beyond &plusmn;2."} Fire{"s" if len(hi_lev1) > 1 else ""}
{", ".join(map(str, hi_lev1))} {"have" if len(hi_lev1) > 1 else "has"} high leverage
(h &gt; 2(k+1)/n = {lev_cut1:.3f}) because {"they lie" if len(hi_lev1) > 1 else "it lies"} at the extremes
of distance. The most influential point is fire {max_cook1} (distance {fires.DISTANCE[max_cook1]} mi,
damage {fires.DAMAGE[max_cook1]}), with Cook's D = {d1.cooks_d.max():.3f}, well below 1. Refitting
without it gives slope {fit1_drop.params['DISTANCE']:.3f} and R&sup2; = {fit1_drop.rsquared:.3f},
so the conclusions do not depend on this fire. No observations were removed.</p>
<h3>1.6 Estimation and prediction</h3>
<p>Table 4 gives 95% confidence intervals for the mean damage and 95% prediction intervals for the
damage of a single new fire at several distances. For example, for a house
3.5 miles from the station, the estimated mean damage is ${row35['Predicted damage']*1000:,.0f};
we are 95% confident the <i>average</i> damage of all such fires is between
${row35['CI mean low']*1000:,.0f} and ${row35['CI mean high']*1000:,.0f}, while the damage of
<i>one</i> such fire is predicted to fall between ${row35['PI low']*1000:,.0f} and
${row35['PI high']*1000:,.0f}. Prediction intervals are wider because they include the variability of
an individual fire around the mean. Intervals are narrowest near the mean distance
({fires.DISTANCE.mean():.2f} mi) and should not be extended beyond 0.7&ndash;6.1 miles.</p>
{html_table(pred1.set_index('Distance (mi)'), 2)}
<p class="caption">Table 4. Estimated mean damage and individual predictions (thousands of dollars).</p>
<h3>1.7 Conclusions for Study 1</h3>
<p>There is very strong evidence of a positive linear association between distance to the nearest
fire station and fire damage. The model is simple, fits well and satisfies its assumptions, so it is
useful for the insurer's pricing decisions within the observed range of distances. Because the data
are observational, the result shows association, not causation: houses far from stations may also
differ in size, construction or response times for other reasons. The sample (15 fires from one
suburb) is small and may not generalize to other areas.</p>""")

# ---- Study 2
R(f"""<h2>2. Study 2: Executive Salaries</h2>
<h3>2.1 Objective and data</h3>
<p>The goal is to find which of ten firm and executive characteristics are associated with
executive salary, and to build a model that predicts salary well. Following the consulting
company's practice, the response is Y = ln(Salary). The sample has n = {n2} executives and no
missing values. X3, X6 and X9 are 0/1 indicator (dummy) variables; the rest are quantitative.
The ID column is only an identifier and is not used.</p>
{html_table(desc2.set_index('Variable'), 2)}
<p class="caption">Table 5. Descriptive statistics, Salaries data.</p>
<h3>2.2 Exploratory analysis</h3>
<p>Log salary is most strongly correlated with experience (r = {corr.loc['Y','X1']:.2f}) and age
(r = {corr.loc['Y','X7']:.2f}), followed by bonus eligibility (r = {corr.loc['Y','X3']:.2f}) and
education, employees supervised and assets (r &asymp; 0.2). Experience and age are highly correlated
with each other (r = {corr.loc['X1','X7']:.2f}), so they carry overlapping information; the
other predictors are nearly uncorrelated with one another (Figure 3). Board membership,
profits, international responsibility and sales show almost no marginal association with
salary (Figure 4).</p>
<img src="{FIG3}" style="width:4.8in">
<p class="caption">Figure 3. Pairwise correlations.</p>
<img src="{FIG4}" style="width:6.5in">
<p class="caption">Figure 4. Log salary versus each candidate predictor (red line: simple LS fit;
boxplots for indicator variables).</p>
<h3>2.3 Full first-order model</h3>
<p>We first fit Y = &beta;<sub>0</sub> + &beta;<sub>1</sub>X1 + &hellip; + &beta;<sub>10</sub>X10 + &epsilon;.
The global F-test of H<sub>0</sub>: &beta;<sub>1</sub> = &hellip; = &beta;<sub>10</sub> = 0 gives
F = {full.fvalue:.2f} on {int(full.df_model)} and {int(full.df_resid)} df ({fmt_p(full.f_pvalue)}), so at
least one predictor is useful. R&sup2; = {full.rsquared:.4f} and adjusted R&sup2; = {full.rsquared_adj:.4f}.
The individual t-tests (Table 6) show that {sig_list} are significant at &alpha; = 0.05, while
{drop_list} are not.</p>
{html_table(coef_full, 4)}
<p class="caption">Table 6. Full model with all ten predictors.</p>
<p><b>Multicollinearity.</b> All variance inflation factors are small (largest: {vif_max_var},
VIF = {vif_max:.2f}; Table 7), far below the usual warning level of 10. The only notable overlap
is between experience and age: age is significant on its own but not once experience is in the
model, meaning age has no additional explanatory value beyond experience.</p>
{html_table(vif.set_index('Variable'), 2)}
<p class="caption">Table 7. Variance inflation factors, full model.</p>
<h3>2.4 Variable selection</h3>
<p><b>Partial (nested) F-test.</b> To test whether the five non-significant predictors can be dropped
together, H<sub>0</sub>: &beta;<sub>6</sub> = &beta;<sub>7</sub> = &beta;<sub>8</sub> = &beta;<sub>9</sub>
= &beta;<sub>10</sub> = 0, we compare the full model with the reduced model containing {sig_list}:
F = {pF_drop:.3f} on 5 and {int(full.df_resid)} df, {fmt_p(pP_drop)}. We do not reject H<sub>0</sub>;
the five variables can be removed without a significant loss of fit. The reduced model has
adjusted R&sup2; = {red.rsquared_adj:.4f}, slightly <i>higher</i> than the full model.</p>
<p><b>All-subsets regression.</b> As a check that does not depend on the order of testing, we fit
all 2<sup>10</sup> &minus; 1 = 1,023 possible first-order models. Figure 5 and Table 8 show the best
model of each size. The highest adjusted R&sup2; occurs at {best_adj_k} predictors, the smallest
model with C<sub>p</sub> &le; p has {best_cp_k} predictors, and BIC is minimized at {best_bic_k}
predictors. {"C<sub>p</sub> and BIC both point to the same five variables (X1, X2, X3, X4, X5)." if best_bic_k == best_cp_k == 5 else "The criteria differ slightly; see Table 8."}
{f"The best {best_adj_k}-variable model ({best.loc[best_adj_k, 'Predictors']}) raises adjusted R&sup2; only from {best.loc[5, 'Adj R2']:.4f} to {best.loc[best_adj_k, 'Adj R2']:.4f}, so the simpler 5-variable model is preferred." if best_adj_k != 5 else ""}</p>
<img src="{FIG5}" style="width:6.5in">
<p class="caption">Figure 5. Best-subsets criteria; the dotted line marks the 5-predictor model.</p>
{html_table(best[['Predictors', 'R2', 'Adj R2', 'Cp', 'BIC']], 4)}
<p class="caption">Table 8. Best model of each size.</p>
<h3>2.5 Refining the model: curvature and interaction</h3>
<p><b>Curvature in experience.</b> Plotting the residuals of the 5-predictor model against each
quantitative predictor (Figure 6) reveals an inverted-U pattern for experience: the first-order
model under-predicts salaries for mid-career executives and over-predicts at both ends. Adding
X1<sup>2</sup> is significant (partial F = {pF_quad:.2f}, {fmt_p(pP_quad)}); the negative coefficient
means salary rises with experience at a <i>decreasing</i> rate.</p>
<img src="{FIG6}" style="width:6.5in">
<p class="caption">Figure 6. Residuals of the 5-predictor model with LOWESS smooth.</p>
<p><b>Interactions with bonus eligibility.</b> It is natural to ask whether the payoff of experience,
education, supervision or assets differs for bonus-eligible executives. Adding all four
interactions X3&middot;X1, X3&middot;X2, X3&middot;X4 and X3&middot;X5 is jointly significant
(F = {pF_int:.2f}, {fmt_p(pP_int)}), but only {", ".join(t.replace(':', '&middot;') for t in keep_int)}
is individually significant. After keeping it, the remaining three are not needed
(F = {pF_rest:.2f}, {fmt_p(pP_rest)}). Figure 7 illustrates the interaction: salaries rise faster
with the number of employees supervised among bonus-eligible executives.</p>
<img src="{FIG7}" style="width:5.4in">
<p class="caption">Figure 7. Fitted lines from the final model for each bonus group.</p>
<h3>2.6 Final model</h3>
<p class="eq" style="font-size:10.5pt">{eq2}</p>
<p>The final model is highly significant (F = {final.fvalue:.2f}, {fmt_p(final.f_pvalue)}) with
R&sup2; = {final.rsquared:.4f}, adjusted R&sup2; = {final.rsquared_adj:.4f} and residual standard error
s<sub>e</sub> = {np.sqrt(final.mse_resid):.4f} on the log scale. {"Every term is significant at the 5% level using both the usual and the heteroskedasticity-robust (HC3) standard errors (Table 9)." if hc3_all_sig and (final.pvalues[1:] < ALPHA).all() else "Table 9 shows the usual and heteroskedasticity-robust (HC3) p-values."}</p>
{html_table(coef_final, 5)}
<p class="caption">Table 9. Final model coefficients, with HC3 robust p-values.</p>
<p><b>Interpretation.</b> Because the response is ln(Salary), a coefficient b corresponds to a
100(e<sup>b</sup> &minus; 1)% change in (median) salary for a one-unit increase, holding the other
variables constant.</p>
<div class="box">
<p>&bull; <b>Education:</b> each additional year is associated with {pct(g['X2']):.2f}% higher salary.<br>
&bull; <b>Corporate assets:</b> each additional $10 million is associated with {pct(10*g['X5']):.2f}% higher salary.<br>
&bull; <b>Experience:</b> the effect diminishes over a career &mdash; one more year adds about
{x1_marg[5]:.1f}% at 5 years, {x1_marg[13]:.1f}% at 13 years and {x1_marg[20]:.1f}% at 20 years. The
curve would peak near {x1_turn:.0f} years, beyond the observed maximum of {sal.X1.max()}, so salary
never actually declines within the data.<br>
&bull; <b>Employees supervised &times; bonus:</b> 100 more employees supervised is associated with
{x4_eff[0]:.1f}% higher salary for executives who are not bonus eligible and
{x4_eff.get(1, float('nan')):.1f}% for those who are. Equivalently, the bonus-eligibility premium
grows with responsibility: about {bonus_eff[100]:.1f}% for an executive supervising 100 employees,
{bonus_eff[int(round(sal.X4.mean()))]:.1f}% at the average of {int(round(sal.X4.mean()))}, and
{bonus_eff[600]:.1f}% at 600.</p>
</div>
{html_table(interp, 3)}
<p class="caption">Table 10. Percentage effects with 95% confidence intervals.</p>
<h3>2.7 Checking the assumptions of the final model</h3>
<img src="{FIG8}" style="width:6.5in">
<p class="caption">Figure 8. Residual diagnostics, final model.</p>
<p><b>Linearity.</b> After adding X1<sup>2</sup> the residuals vs fitted values (Figure 8a) show no
remaining curvature. <b>Normality.</b> The Q-Q plot and histogram (8b, 8c) are consistent with
normal errors (Shapiro&ndash;Wilk W = {sw2.statistic:.4f}, {fmt_p(sw2.pvalue)}).
<b>Independence.</b> The data are a cross-sectional sample of different executives;
Durbin&ndash;Watson d = {dw2:.3f} is close to 2 and Figure 8e shows no pattern.
<b>Constant variance.</b> The Breusch&ndash;Pagan test is significant (LM = {bp2[0]:.2f},
{fmt_p(bp2[1])}). The source is bonus eligibility: the residual standard deviation is
{sd_by_bonus[0]:.3f} for non-eligible executives versus {sd_by_bonus[1]:.3f} for eligible ones
(Figure 8d; BP test on X3 alone, {fmt_p(bp2_x3[1])}). Salaries of executives without a bonus
are therefore less predictable. Since this could make the usual standard errors unreliable, we
recomputed them with HC3 robust standard errors: {"every term remains significant at the 5% level, so the conclusions are unchanged." if hc3_all_sig else "see Table 9 for terms that change."}
The point estimates and predictions are still unbiased.</p>
<p><b>Unusual observations.</b> {f"Only executive {', '.join(map(str, big_res2))} has a studentized residual beyond &plusmn;2.5 ({d2.stud_resid[big_res2[0]]:.2f}), which is not unusual in a sample of 100." if len(big_res2) == 1 else (f"Executives {', '.join(map(str, big_res2))} have studentized residuals beyond &plusmn;2.5." if big_res2 else "No studentized residual exceeds &plusmn;2.5.")}
{len(hi_lev2)} executives have leverage above 2(k+1)/n = {lev_cut2:.2f}. The largest Cook's
distance is {d2.cooks_d.max():.3f} (executive {max_cook2}), far below 1. Refitting without that
executive changes no coefficient by more than {100*max_rel_change:.0f}% and no conclusion, so all
observations were retained.</p>
<h3>2.8 Predictive performance</h3>
<p>Because the model was chosen using the same data, we checked out-of-sample accuracy with
10-fold cross-validation repeated 20 times (Table 11). The final model has the lowest
cross-validated root mean squared error, followed by the 5-predictor model; the full 10-predictor
model predicts worst, confirming that the extra five variables mostly add noise.</p>
{html_table(cv_tab, 4)}
<p class="caption">Table 11. Model comparison (CV RMSE: lower is better).</p>
<p><b>Example.</b> For a bonus-eligible executive with 10 years of experience, 16 years of education,
300 employees supervised and $175 million in corporate assets, the final model predicts
ln(Salary) = {pr2['mean']:.4f}, i.e. a typical (median) salary of about <b>${pred_salary:,.0f}</b>.
The 95% confidence interval for the median salary of all such executives is
${ci_salary[0]:,.0f} to ${ci_salary[1]:,.0f}; the 95% prediction interval for one such executive is
${pi_salary[0]:,.0f} to ${pi_salary[1]:,.0f} (salary units assumed to be dollars; intervals are
obtained by exponentiating the log-scale limits).</p>
<h3>2.9 Conclusions for Study 2</h3>
<p>Executive salary at this firm is driven by experience, education, bonus eligibility, the number of
employees supervised and corporate assets. Experience has the largest effect but with diminishing
returns, and the reward for supervising more people is larger for bonus-eligible executives. Board
membership, age (beyond experience), company profits, international responsibility and total sales
are not associated with salary once these factors are controlled for. The final model explains about
{100*final.rsquared:.0f}% of the variation in log salary and predicts better than the full model.
Limitations: the data come from one firm and are observational, so effects are associations, not
causal; the model was selected and evaluated on the same 100 executives (cross-validation reduces but
does not remove this optimism); and salaries of non-bonus-eligible executives are more variable than
the model's single error variance assumes.</p>""")

# ---- Overall conclusion
R(f"""<h2>3. Overall Conclusions</h2>
<p>Both studies show the value of a complete regression analysis rather than a single fitted
equation. In Study 1, one predictor explains {100*fit1.rsquared:.0f}% of fire damage and all
diagnostic checks support the model. In Study 2, formal tests (partial F), model-selection criteria
(adjusted R&sup2;, C<sub>p</sub>, BIC), residual analysis and cross-validation lead consistently to a
parsimonious model with five predictors, a quadratic term and one interaction. In both cases the
results describe associations in observational data and should be applied only within the range of
the observed predictors.</p>""")

# ---- Appendix: relevant computer output
model_tests = (TAB / "salaries_model_building_tests.txt").read_text(encoding="utf-8")
R(f"""<div class="appendix"><h2>Appendix: Relevant Computer Output</h2>
<h3>A.1 Study 1 &mdash; OLS summary</h3>{pre(fit1.summary().as_text())}
<h3>A.2 Study 2 &mdash; full model</h3>{pre(full.summary().as_text())}
<h3>A.3 Study 2 &mdash; nested model F-tests</h3>{pre(model_tests)}
<h3>A.4 Study 2 &mdash; final model</h3>{pre(final.summary().as_text())}
<h3>A.5 Study 2 &mdash; final model with HC3 robust standard errors</h3>{pre(final_hc3.summary().as_text())}
</div>""")

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>STA 4155 Group Project Report</title><style>{css}</style></head>
<body>{''.join(report)}</body></html>"""
html_path = OUT / "report.html"
html_path.write_text(html, encoding="utf-8")
(OUT / "summary_log.txt").write_text("\n".join(log), encoding="utf-8")


def print_pdf(src, dst):
    """Print the HTML report to PDF with a headless Chromium browser, if available."""
    candidates = [shutil.which(n) for n in ("msedge", "chrome", "google-chrome", "chromium")]
    candidates += [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                   r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                   r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    if dst.exists():
        dst.unlink()                       # never report an old PDF as new
    for exe in filter(None, candidates):
        if not Path(exe).exists():
            continue
        # A throwaway profile keeps the headless browser separate from any open browser window.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
            try:
                subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-first-run",
                                f"--user-data-dir={profile}", "--no-pdf-header-footer",
                                f"--print-to-pdf={dst}", src.as_uri()],
                               check=True, timeout=180, capture_output=True)
            except Exception as exc:
                print(f"  PDF attempt with {exe} failed: {exc}")
            for _ in range(60):            # the file can appear a moment after exit
                if dst.exists() and dst.stat().st_size > 0:
                    time.sleep(1)          # let the browser finish writing
                    return True
                time.sleep(0.5)
    return False


pdf_ok = print_pdf(html_path, OUT / "report.pdf")
say(f"Report written to {html_path}" + (" and report.pdf" if pdf_ok else " (PDF skipped: no browser found)"))
