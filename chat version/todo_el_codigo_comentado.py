# %%
# STA 4155 - TODO EL CODIGO JUNTO, CON EXPLICACIONES EN ESPANOL
# Este archivo es una copia completa y ejecutable del ejemplo.
# No necesitas ejecutar analysis.py antes: aqui se cargan los CSV y se calcula todo.
# Cada marca # %% separa una etapa. VS Code puede mostrarlas como celdas.
# El notebook proyecto_paso_a_paso.ipynb usa exactamente estas mismas celdas.
# Para ejecutarlo como archivo desde la carpeta principal:
# python "chat version/todo_el_codigo_comentado.py"
# Los numeros se calculan; las imagenes son graficos creados por Matplotlib.

from pathlib import Path
import sys
import platform
import importlib.metadata

# Localizar la carpeta que contiene los CSV, tanto en .py como en notebook.
base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next((p for p in [base, *base.parents]
             if (p / "Fires.csv").is_file() and (p / "Salaries.csv").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("Abre el proyecto desde la carpeta sta4155 project")

# Las bibliotecas locales de esta computadora se prepararon para Python 3.14.
# Otros interpretes deben instalar requirements-notebook.txt en su propio entorno.
local_packages = ROOT / ".packages"
if sys.version_info[:2] == (3, 14) and local_packages.exists():
    sys.path.insert(0, str(local_packages))

# pandas maneja tablas; NumPy calcula; statsmodels ajusta las regresiones;
# scipy aporta pruebas estadisticas y Matplotlib dibuja los graficos.
import numpy as np
import pandas as pd
import matplotlib
# En el notebook mostramos las figuras debajo del codigo.
# Como script guardamos las mismas figuras en results, sin abrir ventanas.
try:
    EN_NOTEBOOK = get_ipython().__class__.__name__ == "ZMQInteractiveShell"
except NameError:
    EN_NOTEBOOK = False
if EN_NOTEBOOK:
    get_ipython().run_line_magic("matplotlib", "inline")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro

OUT = ROOT / "chat version" / "results"
OUT.mkdir(exist_ok=True)
report = ["# STA 4155: worked Python example", "Observational associations do not establish causation. Alpha = 0.05. Intervals assume the fitted model is appropriate."]

# Esta funcion muestra tablas en notebook y texto legible al ejecutar el .py.
def mostrar_tabla(tabla):
    if EN_NOTEBOOK:
        from IPython.display import display
        display(tabla)
    else:
        print(tabla.to_string())


def mostrar_figura(fig):
    # La figura se crea con datos y comandos de Python, no leyendo un PNG previo.
    if EN_NOTEBOOK:
        from IPython.display import display
        display(fig)
    plt.close(fig)


# %%
# 1. FUNCION COMPARTIDA PARA LOS MODELOS
# OLS = minimos cuadrados ordinarios: estima los coeficientes que minimizan
# la suma de residuos al cuadrado. La formula indica respuesta ~ predictores.
# Un residuo es el valor observado menos el valor estimado por el modelo.
def analyze(name, data, formula):
    model = smf.ols(formula, data=data, missing="raise").fit()
    (OUT / f"{name}_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")
    # summary() muestra coeficientes, errores estandar, pruebas t, F y R cuadrado.
    # H0 global: todas las pendientes (excepto el intercepto) son cero.
    # Cada prueba t contrasta H0: el coeficiente correspondiente es cero.
    print("\nMODELO:", name, "|", formula)
    print(model.summary())
    # Los intervalos que siguen son los clasicos OLS (suponen varianza constante).
    ci = model.conf_int()
    pd.DataFrame({"estimate": model.params, "SE": model.bse, "p": model.pvalues,
                  "CI_low": ci[0], "CI_high": ci[1]}).to_csv(OUT / f"{name}_coefficients.csv")
    mostrar_tabla(pd.DataFrame({"coef_OLS": model.params,
        "SE_clasico": model.bse, "p_clasico": model.pvalues,
        "IC95_inferior": ci[0], "IC95_superior": ci[1]}))
    # Leverage: posicion inusual de los predictores. Cook: influencia en el ajuste.
    # 4/n es una regla de revision, no una orden de borrar observaciones.
    infl = model.get_influence()
    pd.DataFrame({"row": data.index + 1, "residual": model.resid,
                  "leverage": infl.hat_matrix_diag, "Cooks_D": infl.cooks_distance[0]}).to_csv(OUT / f"{name}_influence.csv", index=False)
    mostrar_tabla(pd.DataFrame({"fila_CSV_sin_encabezado": data.index + 1,
        "residuo": model.resid, "leverage": infl.hat_matrix_diag,
        "Cook_D": infl.cooks_distance[0]}).sort_values("Cook_D", ascending=False).head())
    # BP: H0 = varianza constante. Shapiro: H0 = residuos normales.
    # p < 0.05 aporta evidencia contra H0; p grande no demuestra el supuesto.
    bp = het_breuschpagan(model.resid, model.model.exog)
    sw = shapiro(model.resid)
    mostrar_tabla(pd.DataFrame({"prueba": ["Breusch-Pagan LM", "Shapiro-Wilk"],
        "estadistico": [bp[0], sw.statistic], "p_valor": [bp[1], sw.pvalue]}))
    # Izquierda: buscar curvatura o forma de embudo. Centro: revisar la Q-Q.
    # Derecha: buscar observaciones influyentes. La independencia depende del muestreo.
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].scatter(model.fittedvalues, model.resid)
    ax[0].axhline(0, color="red")
    ax[0].set(xlabel="Fitted", ylabel="Residual", title="Residuals vs fitted")
    sm.qqplot(model.resid, line="s", ax=ax[1])
    ax[2].stem(data.index + 1, infl.cooks_distance[0])
    ax[2].axhline(4/len(data), color="red", linestyle="--")
    ax[2].set(xlabel="Row", ylabel="Cook's D", title="Influence (4/n screening line)")
    fig.tight_layout(); fig.savefig(OUT / f"{name}_diagnostics.png", dpi=160); mostrar_figura(fig)
    report.extend([f"## {name}", f"Formula: `{formula}`. n={int(model.nobs)}; R²={model.rsquared:.4f}; adjusted R²={model.rsquared_adj:.4f}; overall F-test p={model.f_pvalue:.5g}.",
        f"Breusch-Pagan LM p={bp[1]:.5g}; Shapiro-Wilk p={sw.pvalue:.5g}. Small p-values flag possible violations; large p-values do not prove assumptions. Independence depends on sampling design and cannot be established by these tests.",
        f"![Diagnostics]({name}_diagnostics.png)", f"See {name}_summary.txt and coefficient/influence CSV files. Points above 4/n need review, not automatic deletion."])
    # HC3 estima errores estandar robustos a varianza desigual.
    # No cambia las pendientes OLS: cambia su incertidumbre, pruebas e intervalos.
    # No soluciona por si solo una mala forma funcional o falta de independencia.
    robust = model.get_robustcov_results(cov_type="HC3")
    robust_ci = robust.conf_int()
    mostrar_tabla(pd.DataFrame({"coef_OLS": model.params,
        "SE_clasico": model.bse, "p_clasico": model.pvalues,
        "SE_HC3": robust.bse, "p_HC3": robust.pvalues,
        "IC95_HC3_inf": robust_ci[:, 0], "IC95_HC3_sup": robust_ci[:, 1]}))
    (OUT / f"{name}_HC3.txt").write_text(robust.summary().as_text(), encoding="utf-8")
    report.append("Overall hypothesis: all non-intercept slopes equal zero. " + ("Reject at 5%; the model provides evidence of association." if model.f_pvalue < .05 else "Do not reject at 5%; evidence of association is insufficient."))
    report.append("Diagnostic interpretation: " + ("the constant-variance assumption is questioned by the BP test; compare the HC3 output." if bp[1] < .05 else "the BP test does not detect unequal variance at 5%.") + (" The Q-Q plot needs particular attention because Shapiro-Wilk rejects normality." if sw.pvalue < .05 else " Shapiro-Wilk does not reject normality at 5%."))
    return model

# %%
# 2. CARGAR Y REVISAR LOS DATOS ORIGINALES
# read_csv lee los datos. No se modifica ninguno de los tres archivos originales.
# En Salaries, Y YA es log(salario): no hay que aplicarle otro logaritmo.
fires = pd.read_csv(ROOT / "Fires.csv")
salary = pd.read_csv(ROOT / "Salaries.csv")
for label, df in [("fires", fires), ("salaries", salary)]:
    print("\nDATOS:", label, "| filas:", len(df))
    mostrar_tabla(df.head())
    print("Valores faltantes por columna:")
    mostrar_tabla(df.isna().sum().to_frame("faltantes"))
    print("Estadisticas descriptivas:")
    mostrar_tabla(df.describe())
    df.describe().to_csv(OUT / f"{label}_descriptive.csv")
    df.isna().sum().to_csv(OUT / f"{label}_missing.csv")
    if df.isna().any().any():
        raise ValueError(f"Missing data in {label}: decide how to handle before fitting")

# %%
# 3. INCENDIOS: REGRESION LINEAL SIMPLE
# Respuesta: DAMAGE (miles de dolares). Predictor: DISTANCE (millas).
# Pendiente: cambio promedio del dano por una milla adicional.
# R cuadrado: proporcion de variacion del dano explicada dentro de esta muestra.
# No demuestra que distancia cause dano: estos datos son observacionales.
fire = analyze("fires", fires, "DAMAGE ~ DISTANCE")
# %%
# 4. INCENDIOS: PREDICCIONES Y GRAFICO CREADO DESDE CERO
# Solo usamos distancias dentro del rango observado para evitar extrapolar.
# IC de la media: incertidumbre sobre dano promedio a cierta distancia.
# Intervalo de prediccion: incertidumbre sobre un incendio individual (mas ancho).
grid = pd.DataFrame({"DISTANCE": np.linspace(fires.DISTANCE.min(), fires.DISTANCE.max(), 100)})
pred = fire.get_prediction(grid).summary_frame()
pd.concat([grid, pred], axis=1).to_csv(OUT / "fires_predictions.csv", index=False)
mostrar_tabla(pd.concat([grid, pred], axis=1).iloc[[0, 24, 49, 74, 99]])
print(f"Ecuacion estimada: DAMAGE = {fire.params.iloc[0]:.4f} + {fire.params.iloc[1]:.4f} * DISTANCE")
fig, ax = plt.subplots()
ax.scatter(fires.DISTANCE, fires.DAMAGE, label="Observed fires")
ax.plot(grid.DISTANCE, pred["mean"], label="OLS")
ax.fill_between(grid.DISTANCE, pred.mean_ci_lower, pred.mean_ci_upper, alpha=.3, label="95% mean CI")
ax.plot(grid.DISTANCE, pred.obs_ci_lower, "--", color="gray", label="95% individual PI")
ax.plot(grid.DISTANCE, pred.obs_ci_upper, "--", color="gray")
ax.set(xlabel="Distance (miles)", ylabel="Damage (thousands of dollars)")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fires_fit.png", dpi=160); mostrar_figura(fig)
report.extend([f"Estimated damage = {fire.params.iloc[0]:.4f} + {fire.params.iloc[1]:.4f} × distance. One additional mile is associated with {fire.params.iloc[1]:.4f} thousand dollars more damage on average. Slope p={fire.pvalues.iloc[1]:.5g}. The intercept may have little practical meaning if zero miles is outside the data range.", "![Fire model](fires_fit.png)", "Mean confidence bands estimate average damage; wider prediction bands describe one new fire. Avoid extrapolation."])

# %%
# 5. SALARIOS: MODELO MULTIPLE COMPLETO
# ID es un identificador, no un predictor. Y ya esta en logaritmo natural.
# X1 experiencia, X2 educacion, X3 bono (0/1), X4 empleados supervisados,
# X5 activos (millones), X6 junta (0/1), X7 edad, X8 ganancias (millones),
# X9 responsabilidad internacional (0/1), X10 ventas (millones).
# Cada pendiente se interpreta manteniendo constantes los otros predictores.
predictors = [f"X{i}" for i in range(1,11)]
full = analyze("salaries_full", salary, "Y ~ " + " + ".join(predictors))
# %%
# 6. COLINEALIDAD ENTRE PREDICTORES
# VIF mide inflacion de la varianza de un coeficiente por relaciones entre X.
# Un VIF alto invita a investigar; ningun corte sustituye el criterio del curso.
X = full.model.exog
pd.DataFrame({"variable": full.model.exog_names[1:], "VIF": [variance_inflation_factor(X,i) for i in range(1,X.shape[1])]}).to_csv(OUT / "salaries_VIF.csv", index=False)
mostrar_tabla(pd.read_csv(OUT / "salaries_VIF.csv"))

# %%
# 7. REDUCCION EXPLORATORIA POR AIC
# Se prueba quitar una variable cada vez y se conserva el cambio si reduce AIC.
# Menor AIC favorece balance ajuste-complejidad; no garantiza mejor prediccion.
# El modelo completo sigue siendo la referencia. La seleccion usa la misma muestra:
# p-valores e intervalos posteriores no incorporan la incertidumbre de seleccionar.
# Si necesitan evaluar prediccion, hay que usar validacion fuera de esta muestra.
aic_history = [{"paso": 0, "quitada": "ninguna", "AIC": full.aic}]
selected = predictors.copy()
current = full
while len(selected) > 1:
    trials = [(smf.ols("Y ~ " + " + ".join(x for x in selected if x != drop), salary).fit(), drop) for drop in selected]
    candidate, drop = min(trials, key=lambda pair: pair[0].aic)
    if candidate.aic >= current.aic - 1e-8: break
    selected.remove(drop); current = candidate
    aic_history.append({"paso": len(aic_history), "quitada": drop, "AIC": current.aic})
mostrar_tabla(pd.DataFrame(aic_history))
reduced = analyze("salaries_exploratory_AIC", salary, "Y ~ " + " + ".join(selected))
mostrar_tabla(pd.DataFrame({"modelo": ["completo", "AIC exploratorio"],
    "predictores": [len(predictors), len(selected)],
    "R2": [full.rsquared, reduced.rsquared],
    "R2_ajustado": [full.rsquared_adj, reduced.rsquared_adj],
    "AIC": [full.aic, reduced.aic]}))

# %%
# 8. INTERPRETAR COEFICIENTES EN ESCALA DE SALARIO
# 100*(exp(b)-1) convierte un coeficiente logaritmico en cambio porcentual.
# Para una variable binaria, corresponde pasar de 0 a 1, con el resto constante.
# Aqui usamos HC3 porque BP detecta varianza desigual en salarios.
# La interpretacion como mediana supone mediana condicional cero del error log.
# No confundir exp(prediccion log) con salario medio: existe sesgo de transformacion.
hc3 = full.get_robustcov_results(cov_type="HC3")
hc3_ci = hc3.conf_int()
mostrar_tabla(pd.DataFrame({"variable": full.params.index[1:],
    "cambio_porcentual": 100 * np.expm1(full.params.iloc[1:].to_numpy()),
    "IC95_HC3_inf_pct": 100 * np.expm1(hc3_ci[1:, 0]),
    "IC95_HC3_sup_pct": 100 * np.expm1(hc3_ci[1:, 1]),
    "p_HC3": hc3.pvalues[1:]}))

# %%
# 9. GUARDAR EL INFORME AUTOMATICO Y LAS SALIDAS PARA CONSULTARLAS
# El informe original conserva sus intervalos clasicos OLS como referencia.
# Para salarios, comparar con las tablas HC3 mostradas arriba y los archivos HC3.txt.
# Las frases contienen numeros formateados desde los modelos, no resultados escritos a mano.
report.extend(["## Salary interpretation", "Y already contains natural log salary: do not log it again. ID is an identifier, excluded from predictors. X3, X6, X9 are binary indicators. Other coefficients adjust for all included predictors.",
    f"Full AIC={full.aic:.3f}; exploratory reduced AIC={reduced.aic:.3f}. Selected predictors: {', '.join(selected)}. Selection and evaluation use the same sample: reduced-model p-values and intervals ignore selection uncertainty. AIC improvement is not evidence of out-of-sample accuracy.",
    "For each coefficient b, 100*(exp(b)-1) gives the modeled percentage change in median salary for a one-unit increase (or 0-to-1 indicator change), holding other variables fixed. Exponentiating a log prediction estimates median salary, not automatically mean salary. Currency units are not specified in the supplied variable table."])
for term in full.params.index[1:]:
    lo, hi = full.conf_int().loc[term]
    report.append(f"- {term}: adjusted median salary change {100*np.expm1(full.params[term]):.2f}%; 95% CI [{100*np.expm1(lo):.2f}%, {100*np.expm1(hi):.2f}%]; p={full.pvalues[term]:.5g}.")
report.extend(["## Review before writing your own report", "Inspect residual shape, spread, Q-Q departures, VIF and influential cases. Consider robust HC3 standard errors if variance is unequal; they do not fix model misspecification. Check variable relevance and possible nonlinearities against course methods. Neither dataset supports a causal claim without additional design evidence.", "Include your research questions, methods, numerical findings, limitations, relevant computer output, and group contributions. This example does not supply member names or invent contribution percentages."])
(OUT / "REPORT.md").write_text("\n\n".join(report), encoding="utf-8")
print(f"Results saved to {OUT}")


# %%
# 10. REGISTRAR VERSIONES PARA FACILITAR LA REPRODUCCION
# Si otro equipo obtiene un error, comparen estas versiones y su interprete.
versions = {"Python": platform.python_version()}
for package in ["pandas", "numpy", "scipy", "statsmodels", "matplotlib"]:
    versions[package] = importlib.metadata.version(package)
mostrar_tabla(pd.Series(versions, name="version").to_frame())
(OUT / "versiones_ejemplo.txt").write_text(
    "\n".join(f"{k}=={v}" for k, v in versions.items()), encoding="utf-8")
print("Listo: todos los calculos anteriores salieron del codigo y de los CSV.")
