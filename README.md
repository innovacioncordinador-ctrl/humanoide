# Project README — Unitree G1 (técnico)

> Este README proporciona instrucciones técnicas reproducibles para preparar el entorno, instalar dependencias críticas y ejecutar los scripts del repositorio.

## Requisitos exactos (no negociables)

- **Sistema operativo:** Ubuntu 22.04 LTS o 24.04 LTS
- **Python:** 3.11 (recomendado en este archivo)
- **MuJoCo:** 3.1.6 / 3.2.x (ver script y entorno)
- **mujoco_viewer / mujoco-python-viewer:** 0.3.0
- **PyTorch:** según CUDA (en este README aparece `torch==2.9.0` + cu130 como ejemplo)
- **numpy:** >=1.23.5

### Verificaciones rápidas

```bash
lsb_release -a
python --version
nvidia-smi
python -c "import mujoco, mujoco_viewer, torch; print(mujoco.__version__, mujoco_viewer.__version__, torch.__version__)"
```

> Nota: desviaciones en las versiones de `mujoco`, `mujoco_viewer` o la combinación CUDA/PyTorch pueden romper la compatibilidad con el viewer o con cargas JIT (p. ej. `torch.jit.load`). Asegúrese de usar las versiones que su hardware y sus scripts esperan.

## Instalación reproducible (recomendado: conda)

[!tip] Clonar


1. Crear y activar entorno (ejemplo con conda):

```bash
conda create -n g1_udc python=3.11 -y
conda activate g1_udc
pip install --upgrade pip
```

2. Instalar dependencias del proyecto:

```bash
pip install -r requirements.txt
```

3. Instalar MuJoCo (según la versión que use su setup):

```bash
pip install mujoco==3.2.3
```

4. Instalar PyTorch con CUDA (ejemplo para CUDA 13.0/cu130 mostrado en este archivo):

```bash
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu130
```

5. Verificar instalación mínima:

```bash
python -c "import mujoco, mujoco_viewer, torch; print('OK', mujoco.__version__, mujoco_viewer.__version__, torch.__version__)"
```

Si algún import falla, revise las versiones de paquetes y la compatibilidad CUDA/cuDNN de su sistema.

## Estructura relevante del repositorio

- `codigos/` — scripts principales
- `codigos/secuencias_amo/` — secuencias autónomas (módulos independientes)
- `escenas/` — modelos XML para MuJoCo (diferentes configuraciones y DOFs)
- `politicas/` — pesos (`.pt`) usados por las políticas

## Importante — `codigos/secuencias_amo` son módulos autónomos

Los scripts dentro de `codigos/secuencias_amo/` están diseñados para ejecutarse sin depender de la infraestructura completa (`HumanoidEnv`, `adapter`, etc.). Requisitos mínimos:

- Python instalado
- MuJoCo + `mujoco_viewer` operativos
- El fichero XML de la escena correspondiente (ya incluido en `escenas/`)
- Opcional: si una secuencia requiere una política preentrenada, colocar el `.pt` correspondiente en `politicas/` y actualizar la ruta en el script

### Comandos de ejecución independientes (ejemplos)

```bash
# Secuencia completa autónoma (usa los módulos en secuencias_amo)
python codigos/secuencias_amo/secuencia_completa.py

# Saludo autónomo (versión limpia)
python codigos/secuencias_amo/saludo.py
```

Estos scripts contienen controladores autocontenidos (p. ej. `PickAndPlaceController`) y rutinas de interpolación, por lo que no requieren `HumanoidEnv` ni el adapter para funcionar.

## Scripts que requieren repositorios externos (HumanoidEnv completo)

Algunos scripts en la raíz (`codigos/g1_caja.py`, `codigos/safe.py`, etc.) asumen que los repositorios oficiales de Unitree (o forks compatibles) están disponibles en rutas conocidas. Ejemplos:

- https://github.com/unitreerobotics/unitree_mujoco
- https://github.com/unitreerobotics/unitree_sdk2_python
- https://github.com/unitreerobotics/unitree_sdk2
- https://github.com/unitreerobotics/unitree_robots

Ejemplo de estructura necesaria (sugerida):

```
~/unitree_mujoco/
~/unitree_robots/g1/
~/tu-proyecto-g1/                  ← este repositorio
    ├── codigos/g1_caja.py
    ├── escenas/interaccion.xml     ← puede provenir de unitree_mujoco
    └── politicas/test.pt           ← puede provenir de otro repo
```

Ejemplos de ejecución (requieren los repos externos configurados):

```bash
python codigos/g1_caja.py      # carga interaccion.xml desde unitree_mujoco
python codigos/safe.py
python codigos/test_caja.py
python codigos/saludo.py       # el de la raíz
```

Si no tiene los repositorios oficiales en las rutas esperadas, estos scripts fallarán; edite las rutas dentro de los scripts o coloque los repositorios en las rutas sugeridas.

## Modelos XML (resumen técnico)

- `g1_23dof.xml` — 23 DOF, modelo reducido (más estable)
- `g1_29dof.xml` — 29 DOF, manos de 3 dedos (experimental)
- `g1_29dof_tables.xml` — escena con mesas + caja (pick & place realista)
- `interaccion.xml` — escena pública usada en demos (mesa baja + caja)
- `scene_23dof.xml` — plano vacío, máxima estabilidad

El script que ejecute debe apuntar al XML adecuado; las rutas por defecto están configuradas en los scripts bajo `codigos/`.

## Notas y limitaciones técnicas (rápido)

- El agarre real dinámico no está soportado por limitaciones del modelo MuJoCo (se usa un truco kinemático).
- Evite combinar `freejoint` en objetos con múltiples caminos al ground (puede producir NaNs en qpos).
- No hay percepción visual robusta en la simulación (render real ≠ render sim).

## Comandos recomendados para desarrollo y debugging

```bash
# Activar entorno
conda activate g1-env

# Ejecutar una secuencia autónoma
python codigos/secuencias_amo/secuencia_completa.py

# Ejecutar demo completa que usa HumanoidEnv + adapter
python codigos/g1_caja.py
```

---

