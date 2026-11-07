# claude version — STA 4155 Group Project

Un solo script (`analysis.py`) hace TODO: lee los dos CSV de la carpeta de arriba,
corre los análisis, guarda gráficas y tablas, y escribe el reporte final en inglés
(HTML + PDF, tamaño carta, doble espacio, letra de 12 pt).

```powershell
python "claude version/analysis.py"
```

Tarda ~20 s. Resultado:

| Archivo | Qué es |
|---|---|
| `output/report.pdf` | **El reporte para entregar** (20 págs). Falta llenar la tabla de miembros y % en la portada. |
| `output/report.html` | El mismo reporte (se abre en el navegador; desde ahí también puedes imprimirlo a PDF). |
| `output/figures/` | Las 8 figuras (PNG). |
| `output/tables/` | Todas las tablas en CSV y las salidas crudas de statsmodels (`*_summary.txt`). |
| `analysis.py` | El código que se entrega por separado (el PDF lo pide así). |

---

## Cómo está organizada la versión de Chat (para comparar)

| Archivo de `chat version/` | Qué hace |
|---|---|
| `analysis.py` | El análisis principal. Tiene una función `analyze()` que ajusta cualquier modelo, guarda el summary, coeficientes, influencia, prueba Breusch–Pagan y Shapiro–Wilk, y dibuja 3 gráficas de diagnóstico. La llama 3 veces: Fires, Salaries completo y Salaries reducido por AIC. Escribe `results/REPORT.md`. |
| `todo_el_codigo_comentado.py` | Lo mismo pero con comentarios en español, para estudiar. |
| `preparar_notebook.py` / `ejecutar_notebook.py` | Convierten el script en el notebook `proyecto_paso_a_paso.ipynb` y lo ejecutan. Solo son utilidades. |
| `results/` | Las salidas. `REPORT.md` es un **borrador**, no el reporte final. |

Su enfoque en Salaries: modelo completo con las 10 variables → eliminación hacia atrás por AIC
→ reporta errores HC3 porque la varianza no es constante.

## Qué hice diferente en mi versión

1. **Reporte terminado, no borrador.** Genera el documento formal con portada, resumen ejecutivo,
   secciones, tablas numeradas, figuras con pie, interpretaciones en palabras y apéndice con la
   salida de computadora (el PDF pide incluir la salida relevante). Todos los números del texto
   se insertan desde el código (el PDF prohíbe cálculos a mano).
2. **Sigue los métodos del curso** (vi tus notas de R de los capítulos 4, 16 y 17): correlación,
   recta de mínimos cuadrados, prueba t de la pendiente, IC de la pendiente, tabla ANOVA, R², gráfica
   de residuos vs ajustados y vs orden, Q-Q, Durbin–Watson, bandas de confianza y de predicción.
3. **Salaries: un modelo mejor que el de Chat.** Chat se queda con el modelo lineal. Yo encontré dos
   cosas en las gráficas de residuos:
   - **Curvatura en experiencia (X1)**: los residuos forman una U invertida → se agrega X1². El sueldo
     sube con la experiencia pero cada año extra vale menos (3.8% a los 5 años, 1.8% a los 20).
   - **Interacción bono × empleados supervisados (X3·X4)**: supervisar a 100 personas más sube el sueldo
     3.3% si no tienes bono y 6.5% si sí lo tienes.
   - Modelo final: adj R² = 0.936 (vs 0.914 del completo).
4. **Selección de variables justificada de 3 formas**: prueba F parcial (quitar X6–X10 juntas, p = 0.756),
   *best subsets* de los 1,023 modelos posibles (adj R², Cp, BIC) y validación cruzada 10-fold.
   Todos apuntan a X1, X2, X3, X4, X5.
5. **Explico el origen de la varianza no constante**: viene del bono (los que no tienen bono tienen
   sueldos más dispersos). Se reportan p-values HC3 y todo sigue siendo significativo.
6. **Análisis de sensibilidad**: se reajusta sin la observación más influyente (fuego 13, ejecutivo 40)
   y las conclusiones no cambian.

## Mapa del código (`analysis.py`)

| Sección | Qué hace | Salida |
|---|---|---|
| **0. Setup** | Rutas, librerías (usa `../.packages` si existe), estilo de gráficas. | — |
| **Helpers** | `coef_table` (tabla tipo `summary(lm)` de R), `anova_slr` (tabla ANOVA), `diagnostics` (residuos estudentizados, leverage, Cook). | — |
| **1.1** Exploratorio | Estadística descriptiva y correlación r (Cap. 4). | Tabla 1 |
| **1.2** Ajuste | `DAMAGE ~ DISTANCE`, coeficientes, IC, ANOVA (Cap. 16). | Tablas 2–3 |
| **1.3** Supuestos | Shapiro–Wilk, Breusch–Pagan, Durbin–Watson, leverage, Cook, reajuste sin fuego 13 (Cap. 17). | Fig. 2 |
| **1.4** Predicción | IC de la media y PI individual a 1, 2, 3.5, 5, 6 millas. | Tabla 4, Fig. 1 |
| **2.1** Exploratorio | Descriptivos, matriz de correlación, Y vs cada X. | Figs. 3–4 |
| **2.2** Modelo completo | 10 predictores, prueba F global, pruebas t, VIF (Cap. 18). | Tablas 6–7 |
| **2.3** Selección | F parcial + best subsets (Cap. 19). | Fig. 5, Tabla 8 |
| **2.4** Refinamiento | Residuos vs cada X → X1²; bloque de interacciones con X3 → se queda X3·X4. | Figs. 6–7 |
| **2.5** Modelo final | Coeficientes + HC3, interpretación en % (`100·(e^b − 1)`), diagnósticos. | Tablas 9–10, Fig. 8 |
| **2.6** Validación | CV 10-fold repetida 20 veces: completo vs reducido vs final. | Tabla 11 |
| **2.7** Predicción | Sueldo esperado de un ejecutivo típico con IC y PI (en dólares). | — |
| **REPORT** | Arma el HTML con todo el texto y lo imprime a PDF con Edge/Chrome. | `report.html/.pdf` |

## Resultados clave

**Estudio 1 (incendios):** Daño = 10.278 + 4.919 × Distancia. Cada milla extra ≈ $4,919 más de daño
(IC 95%: $4,071–$5,768). R² = 0.924. Se cumplen todos los supuestos.

**Estudio 2 (salarios):** ln(Salario) depende de experiencia (con rendimientos decrecientes), educación
(+3.1% por año), bono, empleados supervisados (más fuerte si hay bono) y activos (+2.4% por cada $10M).
Junta directiva, edad, ganancias, responsabilidad internacional y ventas no aportan.

## Lo que falta antes de entregar

- Llenar la tabla de **miembros del grupo y % de contribución** en la portada (en `analysis.py`,
  sección "Title page", o directamente en el documento).
- Leer el reporte y ajustar la redacción a su estilo: el PDF pide trabajo independiente entre grupos.
