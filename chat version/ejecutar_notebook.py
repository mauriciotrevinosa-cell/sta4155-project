"""Regenera el notebook desde el .py comentado y calcula todas sus salidas."""
from pathlib import Path
import json
import os
import runpy
import sys

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parent
PACKAGES = ROOT / ".packages"
if sys.version_info[:2] == (3, 14) and PACKAGES.exists():
    sys.path.insert(0, str(PACKAGES))
    os.environ["PYTHONPATH"] = str(PACKAGES) + os.pathsep + os.environ.get("PYTHONPATH", "")

print("Cargando las bibliotecas de Jupyter...", flush=True)
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

# Configuracion local: no instala ni cambia kernels del sistema.
kernel_root = ROOT / ".notebook-runtime" / "kernels"
kernel_folder = kernel_root / "sta4155-local"
kernel_folder.mkdir(parents=True, exist_ok=True)
(kernel_folder / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "STA 4155 local", "language": "python",
}), encoding="utf-8")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".notebook-runtime")

# El generador copia solo el codigo. nbclient ejecuta cada celda en un kernel real.
runpy.run_path(str(FOLDER / "preparar_notebook.py"), run_name="__main__")
path = FOLDER / "proyecto_paso_a_paso.ipynb"
notebook = nbformat.read(path, as_version=4)
specs = KernelSpecManager(kernel_dirs=[str(kernel_root)])
manager = KernelManager(kernel_name="sta4155-local", kernel_spec_manager=specs)
print("Ejecutando las celdas en un kernel de Python...", flush=True)
NotebookClient(
    notebook, km=manager, timeout=180, startup_timeout=60, allow_errors=False,
    resources={"metadata": {"path": str(FOLDER)}},
).execute()
nbformat.validate(notebook)
nbformat.write(notebook, path)
executed = [c for c in notebook.cells if c.cell_type == "code"]
print(f"Verificado: {len(executed)} celdas ejecutadas sin errores. {path}")
