"""Build the learner notebook from the single commented Python source.

Run after editing todo_el_codigo_comentado.py, then execute the notebook.
This tool copies code cells; it never invents or copies saved results.
"""
from pathlib import Path
import json
import re
import uuid

folder = Path(__file__).resolve().parent
source = (folder / "todo_el_codigo_comentado.py").read_text(encoding="utf-8")
cells = []
for section in re.split(r"^# %%\s*$", source, flags=re.MULTILINE):
    if not section.strip():
        continue
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {},
        "id": uuid.uuid4().hex[:8], "outputs": [],
        "source": section.strip() + "\n",
    })
cells.insert(0, {
    "cell_type": "markdown", "metadata": {}, "id": "intro",
    "source": "# STA 4155: codigo y resultados paso a paso\n\n"
              "Cada celda contiene codigo real del archivo **todo_el_codigo_comentado.py**. "
              "Las tablas y los graficos debajo se producen al ejecutar ese codigo desde los CSV originales. "
              "No se cargan fotos de resultados anteriores.\n\n"
              "Lee los comentarios en espanol. Puedes ejecutar una celda con **Shift+Enter** "
              "o usar **Run All** para ejecutar todo en orden. Los resultados guardados "
              "permiten revisar el ejemplo sin ejecutarlo de nuevo.\n\n"
              "El informe para entregar es un documento separado: revisen el formato del PDF, "
              "expliquen las decisiones con sus palabras e incluyan las contribuciones del grupo.",
})
notebook = {
    "cells": cells, "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
}
target = folder / "proyecto_paso_a_paso.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook creado sin salidas: {target}")
