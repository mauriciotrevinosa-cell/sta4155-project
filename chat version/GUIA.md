# Como estudiar este ejemplo

La forma mas directa ahora es abrir `proyecto_paso_a_paso.ipynb`: muestra
codigo y salidas juntas. El archivo `todo_el_codigo_comentado.py` contiene
TODO el analisis en una sola pieza con comentarios en espanol; se ejecuta
por si solo y usa el mismo codigo que las celdas del notebook.

Los graficos se crean con Matplotlib desde los datos. No se cargan imagenes
anteriores para sustituir calculos. Las salidas guardadas sirven para revisar
una ejecucion ya realizada.

En salarios, compara las tablas clasicas OLS con las tablas HC3: el ejemplo
detecta varianza desigual y muestra las dos formas de medir incertidumbre.

1. Ejecuta analysis.py y abre results/REPORT.md con Ctrl+Shift+V.
2. Lee los archivos summary.txt: coeficientes, errores estandar, pruebas t, F, R cuadrado e intervalos.
3. Compara los graficos con los supuestos: linealidad, varianza constante, normalidad aproximada y observaciones influyentes. La independencia depende del muestreo.
4. Revisa VIF para colinealidad en Salaries. El modelo reducido por AIC es exploratorio: no es automaticamente el mejor para predecir.
5. Escribe sus conclusiones en sus palabras usando los numeros producidos, e incluye las limitaciones.

## Diccionario de Salaries
Y: logaritmo natural del salario (ya transformado).
X1: experiencia en anos. X2: educacion en anos. X3: elegibilidad para bono (1 si).
X4: empleados supervisados. X5: activos corporativos en millones de dolares.
X6: miembro de junta (1 si). X7: edad en anos.
X8: ganancias en los ultimos 12 meses en millones de dolares.
X9: responsabilidad internacional (1 si). X10: ventas en los ultimos 12 meses en millones de dolares.
ID no se usa como predictor.

## Organizacion del informe del grupo
Introduccion y preguntas; datos y variables; metodos; estudio de incendios; estudio de salarios; diagnosticos y limitaciones; conclusiones; salidas relevantes; miembros y contribuciones.

El ejemplo conserva el modelo completo como referencia y muestra una reduccion por AIC para discutir. Revisen alternativas justificadas por el curso. No se eliminan observaciones por su influencia de forma automatica. Para comparar capacidad predictiva se necesita validacion fuera de la muestra utilizada para seleccionar variables.
