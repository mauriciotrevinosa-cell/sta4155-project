"""Worked learning example. Run from any directory; originals are never modified."""
from pathlib import Path
import sys
# Optional local libraries downloaded for this example.
local_packages = Path(__file__).resolve().parents[1] / ".packages"
if local_packages.exists():
    sys.path.insert(0, str(local_packages))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
report = ["# STA 4155: worked Python example", "Observational associations do not establish causation. Alpha = 0.05. Intervals assume the fitted model is appropriate."]

def analyze(name, data, formula):
    model = smf.ols(formula, data=data, missing="raise").fit()
    (OUT / f"{name}_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")
    ci = model.conf_int()
    pd.DataFrame({"estimate": model.params, "SE": model.bse, "p": model.pvalues,
                  "CI_low": ci[0], "CI_high": ci[1]}).to_csv(OUT / f"{name}_coefficients.csv")
    infl = model.get_influence()
    pd.DataFrame({"row": data.index + 1, "residual": model.resid,
                  "leverage": infl.hat_matrix_diag, "Cooks_D": infl.cooks_distance[0]}).to_csv(OUT / f"{name}_influence.csv", index=False)
    bp = het_breuschpagan(model.resid, model.model.exog)
    sw = shapiro(model.resid)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].scatter(model.fittedvalues, model.resid)
    ax[0].axhline(0, color="red")
    ax[0].set(xlabel="Fitted", ylabel="Residual", title="Residuals vs fitted")
    sm.qqplot(model.resid, line="s", ax=ax[1])
    ax[2].stem(data.index + 1, infl.cooks_distance[0])
    ax[2].axhline(4/len(data), color="red", linestyle="--")
    ax[2].set(xlabel="Row", ylabel="Cook's D", title="Influence (4/n screening line)")
    fig.tight_layout(); fig.savefig(OUT / f"{name}_diagnostics.png", dpi=160); plt.close(fig)
    report.extend([f"## {name}", f"Formula: `{formula}`. n={int(model.nobs)}; R²={model.rsquared:.4f}; adjusted R²={model.rsquared_adj:.4f}; overall F-test p={model.f_pvalue:.5g}.",
        f"Breusch-Pagan LM p={bp[1]:.5g}; Shapiro-Wilk p={sw.pvalue:.5g}. Small p-values flag possible violations; large p-values do not prove assumptions. Independence depends on sampling design and cannot be established by these tests.",
        f"![Diagnostics]({name}_diagnostics.png)", f"See {name}_summary.txt and coefficient/influence CSV files. Points above 4/n need review, not automatic deletion."])
    robust = model.get_robustcov_results(cov_type="HC3")
    (OUT / f"{name}_HC3.txt").write_text(robust.summary().as_text(), encoding="utf-8")
    report.append("Overall hypothesis: all non-intercept slopes equal zero. " + ("Reject at 5%; the model provides evidence of association." if model.f_pvalue < .05 else "Do not reject at 5%; evidence of association is insufficient."))
    report.append("Diagnostic interpretation: " + ("the constant-variance assumption is questioned by the BP test; compare the HC3 output." if bp[1] < .05 else "the BP test does not detect unequal variance at 5%.") + (" The Q-Q plot needs particular attention because Shapiro-Wilk rejects normality." if sw.pvalue < .05 else " Shapiro-Wilk does not reject normality at 5%."))
    return model

fires = pd.read_csv(ROOT / "Fires.csv")
salary = pd.read_csv(ROOT / "Salaries.csv")
for label, df in [("fires", fires), ("salaries", salary)]:
    df.describe().to_csv(OUT / f"{label}_descriptive.csv")
    df.isna().sum().to_csv(OUT / f"{label}_missing.csv")
    if df.isna().any().any():
        raise ValueError(f"Missing data in {label}: decide how to handle before fitting")

fire = analyze("fires", fires, "DAMAGE ~ DISTANCE")
grid = pd.DataFrame({"DISTANCE": np.linspace(fires.DISTANCE.min(), fires.DISTANCE.max(), 100)})
pred = fire.get_prediction(grid).summary_frame()
pd.concat([grid, pred], axis=1).to_csv(OUT / "fires_predictions.csv", index=False)
fig, ax = plt.subplots()
ax.scatter(fires.DISTANCE, fires.DAMAGE, label="Observed fires")
ax.plot(grid.DISTANCE, pred["mean"], label="OLS")
ax.fill_between(grid.DISTANCE, pred.mean_ci_lower, pred.mean_ci_upper, alpha=.3, label="95% mean CI")
ax.plot(grid.DISTANCE, pred.obs_ci_lower, "--", color="gray", label="95% individual PI")
ax.plot(grid.DISTANCE, pred.obs_ci_upper, "--", color="gray")
ax.set(xlabel="Distance (miles)", ylabel="Damage (thousands of dollars)")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fires_fit.png", dpi=160); plt.close(fig)
report.extend([f"Estimated damage = {fire.params.iloc[0]:.4f} + {fire.params.iloc[1]:.4f} × distance. One additional mile is associated with {fire.params.iloc[1]:.4f} thousand dollars more damage on average. Slope p={fire.pvalues.iloc[1]:.5g}. The intercept may have little practical meaning if zero miles is outside the data range.", "![Fire model](fires_fit.png)", "Mean confidence bands estimate average damage; wider prediction bands describe one new fire. Avoid extrapolation."])

predictors = [f"X{i}" for i in range(1,11)]
full = analyze("salaries_full", salary, "Y ~ " + " + ".join(predictors))
X = full.model.exog
pd.DataFrame({"variable": full.model.exog_names[1:], "VIF": [variance_inflation_factor(X,i) for i in range(1,X.shape[1])]}).to_csv(OUT / "salaries_VIF.csv", index=False)
# Exploratory backward AIC: retain full model for primary inference.
selected = predictors.copy()
current = full
while len(selected) > 1:
    trials = [(smf.ols("Y ~ " + " + ".join(x for x in selected if x != drop), salary).fit(), drop) for drop in selected]
    candidate, drop = min(trials, key=lambda pair: pair[0].aic)
    if candidate.aic >= current.aic - 1e-8: break
    selected.remove(drop); current = candidate
reduced = analyze("salaries_exploratory_AIC", salary, "Y ~ " + " + ".join(selected))
report.extend(["## Salary interpretation", "Y already contains natural log salary: do not log it again. ID is an identifier, excluded from predictors. X3, X6, X9 are binary indicators. Other coefficients adjust for all included predictors.",
    f"Full AIC={full.aic:.3f}; exploratory reduced AIC={reduced.aic:.3f}. Selected predictors: {', '.join(selected)}. Selection and evaluation use the same sample: reduced-model p-values and intervals ignore selection uncertainty. AIC improvement is not evidence of out-of-sample accuracy.",
    "For each coefficient b, 100*(exp(b)-1) gives the modeled percentage change in median salary for a one-unit increase (or 0-to-1 indicator change), holding other variables fixed. Exponentiating a log prediction estimates median salary, not automatically mean salary. Currency units are not specified in the supplied variable table."])
for term in full.params.index[1:]:
    lo, hi = full.conf_int().loc[term]
    report.append(f"- {term}: adjusted median salary change {100*np.expm1(full.params[term]):.2f}%; 95% CI [{100*np.expm1(lo):.2f}%, {100*np.expm1(hi):.2f}%]; p={full.pvalues[term]:.5g}.")
report.extend(["## Review before writing your own report", "Inspect residual shape, spread, Q-Q departures, VIF and influential cases. Consider robust HC3 standard errors if variance is unequal; they do not fix model misspecification. Check variable relevance and possible nonlinearities against course methods. Neither dataset supports a causal claim without additional design evidence.", "Include your research questions, methods, numerical findings, limitations, relevant computer output, and group contributions. This example does not supply member names or invent contribution percentages."])
(OUT / "REPORT.md").write_text("\n\n".join(report), encoding="utf-8")
print(f"Results saved to {OUT}")
