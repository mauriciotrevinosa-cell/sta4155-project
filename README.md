# STA 4155 - Proyecto en Python

Abre esta carpeta completa en VS Code: File > Open Folder.

## Ver el codigo junto con sus resultados

1. Abre `chat version/proyecto_paso_a_paso.ipynb`. Cada bloque contiene codigo
   Python y debajo aparecen sus tablas, pruebas y graficos calculados.
2. Para revisar TODO el codigo en un solo archivo, abre
   `chat version/todo_el_codigo_comentado.py`. Tiene comentarios en espanol y
   es independiente: carga los CSV y hace el analisis completo por si solo.
3. `chat version/results/ejecucion_completa.txt` muestra la salida de una
   ejecucion del archivo comentado como script.

Los PNG son exportaciones de graficos dibujados por Python. No son fotos
usadas como entrada. Los resultados numericos se calculan desde los CSV.

Para volver a ejecutar el archivo unico desde la carpeta principal:

```powershell
python "chat version/todo_el_codigo_comentado.py"
```

Para ejecutar las celdas en VS Code, selecciona un kernel de Python y pulsa
**Run All**, o **Shift+Enter** para avanzar celda por celda. Para preparar un
entorno nuevo, usa los comandos de abajo sustituyendo `requirements.txt`
por `requirements-notebook.txt`.

Si cambias el archivo comentado, `chat version/ejecutar_notebook.py` vuelve a
crear y ejecutar el notebook a partir del mismo codigo. No necesitas estos
archivos auxiliares para ejecutar el analisis del archivo unico.

- `chat version`: ejemplo reproducible con resultados y explicaciones.
- `claude version`: espacio reservado para la version de Claude.
- `owers version`: espacio para construir su propia version.
- Los tres archivos originales estan aqui, sin modificaciones.

## Ejecutar
En la terminal de VS Code, desde esta carpeta:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python "chat version/analysis.py"
```

Selecciona .venv como interprete de Python en VS Code. Los resultados se guardan en `chat version/results`; abre REPORT.md con la vista previa de Markdown.

El ejemplo es una base de aprendizaje. Revisen los supuestos, la seleccion de variables y las interpretaciones, y redacten su propia version. El PDF pide un informe a doble espacio, papel carta, fuente minima 11, salidas relevantes, codigo separado y miembros con porcentajes de contribucion. El Markdown generado es un borrador de contenido, no el documento final con ese formato.
