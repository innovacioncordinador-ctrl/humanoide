Project README — Unitree G1 (técnico)

Este README contiene instrucciones técnicas reproducibles para preparar el entorno, instalar dependencias críticas y ejecutar los scripts del repositorio.

**Requisitos exactos (no negociables)**

- **Sistema operativo:** Ubuntu 22.04 LTS o 24.04 LTS
- **Python:** 3.11 (recomendado)
- **MuJoCo:** 3.1.6
- **mujoco-python-viewer:** 0.3.0
- **PyTorch:** 2.9.0 cu130 (si usa CUDA 12.1) — usar la rueda oficial
- **numpy:** >=1.23.5

Verificaciones rápidas:

```bash
lsb_release -a
python --version
nvidia-smi
python -c "import mujoco, mujoco_viewer, torch; print(mujoco.__version__, mujoco_viewer.__version__, torch.__version__)"
```

Nota: desviaciones en las versiones de `mujoco`, `mujoco_viewer` o la combinación CUDA/PyTorch pueden romper la compatibilidad con el viewer o con cargas JIT (p. ej. `torch.jit.load`).

**Instalación reproducible (recomendado: conda)**

1) Crear y activar entorno (ejemplo con conda):

```bash
conda create -n g1-env python=3.10.12 -y
conda activate g1-env
pip install --upgrade pip
```

1) Instalar dependencias del proyecto:

```bash
pip install -r requirements.txt
```

1) Instalar MuJoCo y viewer (versiones exactas):

```bash
pip install mujoco==3.1.6
pip install mujoco_viewer==0.3.0
```

1) Instalar PyTorch con CUDA (si dispone de GPU compatible). Ejemplo para CUDA 12.1:

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

1) Verificar instalación mínima:

```bash
python -c "import mujoco, mujoco_viewer, torch; print('OK', mujoco.__version__, mujoco_viewer.__version__, torch.__version__)"
```

Si algún import falla, repase las versiones de paquetes y la compatibilidad CUDA/cuDNN de su sistema.

**Estructura relevante del repositorio**

- `codigos/` — scripts principales.
- `codigos/secuencias_amo/` — secuencias autónomas (módulos independientes).
- `escenas/` — modelos XML para MuJoCo (diferentes configuraciones y DOFs).
- `politicas/` — pesos (`.pt`) usados por las políticas.

**Importante — `codigos/secuencias_amo` son módulos autónomos**

Los scripts dentro de `codigos/secuencias_amo/` están diseñados para ejecutarse sin depender de la infraestructura completa (HumanoidEnv, adapter, etc.). Requisitos mínimos:

- Python instalado
- MuJoCo + `mujoco_viewer` operativos
- El fichero XML de la escena correspondiente (ya incluido en `escenas/`)
- Opcional: si una secuencia requiere una política preentrenada, poner el `.pt` correspondiente en `politicas/` y actualizar la ruta en el script.

Comandos de ejecución independientes (ejemplos):

```bash
# Secuencia completa autónoma (usa los módulos en secuencias_amo)
python codigos/secuencias_amo/secuencia_completa.py

# Saludo autónomo (versión limpia)
python codigos/secuencias_amo/saludo.py
```

Estos scripts contienen controladores autocontenidos (p. ej. `PickAndPlaceController`) y rutinas de interpolación, por lo que no requieren `HumanoidEnv` ni el adapter para funcionar.

**Scripts que requieren el entorno completo**

Los scripts en la raíz `codigos/` (por ejemplo, `g1_caja.py`, `safe.py`, `test_caja.py`, `saludo.py`) usan la arquitectura completa (HumanoidEnv + adapter + política cargada). Para ejecutarlos haga:

```bash
conda activate g1-env
python codigos/g1_caja.py   # usa interaccion.xml y política completa
python codigos/safe.py      # usa scene_23dof.xml (versión segura)
```

Compruebe y edite las rutas de `politicas/*.pt` dentro de cada script si los modelos no están en el lugar esperado.

**Modelos XML (resumen técnico)**

- `g1_23dof.xml` — 23 DOF, modelo reducido (más estable)
- `g1_29dof.xml` — 29 DOF, manos de 3 dedos (experimental)
- `g1_29dof_tables.xml` — escena con mesas + caja (pick & place realista)
- `interaccion.xml` — escena pública usada en demos (mesa baja + caja)
- `scene_23dof.xml` — plano vacío, máxima estabilidad

El script que ejecute debe apuntar al XML adecuado; las rutas por defecto están configuradas en los scripts bajo `codigos/`.

**Notas y limitaciones técnicas (rápido)**

- El agarre real dinámico no está soportado por limitaciones del modelo MuJoCo (se usa un truco kinemático).
- Evitar combinar `freejoint` en objetos con múltiples caminos al ground (puede producir NaNs en qpos).
- No hay percepción visual robusta en la simulación (render real ≠ render sim).

**Comandos recomendados para desarrollo y debugging**

```bash
# Activar entorno
conda activate g1-env

# Ejecutar una secuencia autónoma
python codigos/secuencias_amo/secuencia_completa.py

# Ejecutar demo completa que usa HumanoidEnv + adapter
python codigos/g1_caja.py
```

Si desea, puedo:

- ejecutar una comprobación rápida de los imports en su entorno actual (si me lo permite),
- o añadir un README más corto en `codigos/secuencias_amo/` con ejemplos de parámetros por script.

---

Archivo actualizado: [README.md](README.md)
