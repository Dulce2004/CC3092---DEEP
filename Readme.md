# Laboratorio 9: Explainable AI (XAI)

Este laboratorio estudia técnicas de **Inteligencia Artificial Explicable (XAI)** aplicadas a modelos de visión por computadora y aprendizaje automático tabular. El cuaderno [`Lab 9.ipynb`](./Lab%209.ipynb) combina implementaciones desde cero, verificaciones matemáticas y experimentos de evaluación para analizar qué aprende un modelo y por qué produce una predicción determinada.

## Objetivos

- Comprender el problema de la caja negra en modelos de aprendizaje profundo.
- Generar explicaciones visuales para una `ResNet-18` mediante **Grad-CAM** y **CAM**.
- Analizar decisiones de riesgo crediticio con **SHAP**.
- Implementar **LIME tabular** y la fórmula de Shapley desde cero.
- Generar explicaciones contrafactuales para modificar una decisión.
- Validar la fidelidad y estabilidad de las explicaciones mediante pruebas causales y de aleatorización.

## Requisitos

- Python 3.10 o superior.
- Jupyter Notebook o JupyterLab.
- Conexión a internet para descargar los pesos de ImageNet, las imágenes de ejemplo y el conjunto de datos German Credit.
- Un entorno con capacidad suficiente para ejecutar PyTorch, `shap` y los experimentos de muestreo.

### Instalación

Se recomienda crear un entorno virtual antes de instalar las dependencias:

```bash
python -m venv .venv
```

En Windows:

```powershell
.venv\Scripts\Activate.ps1
```

En macOS/Linux:

```bash
source .venv/bin/activate
```

Instalar las dependencias:

```bash
pip install torch torchvision pillow numpy pandas matplotlib scikit-learn scipy shap jupyter
```

## Ejecución

1. Clonar o abrir este repositorio.
2. Activar el entorno virtual e instalar las dependencias.
3. Abrir el cuaderno:

   ```bash
   jupyter notebook "Lab 9.ipynb"
   ```

4. Ejecutar las celdas en orden. Varias secciones reutilizan el modelo, los tensores, el conjunto de datos y los resultados calculados en celdas anteriores.

La semilla aleatoria utilizada en los experimentos principales es `42`. Las ejecuciones pueden tardar varios minutos, especialmente las comparaciones de `KernelSHAP`, la estabilidad de LIME y la generación de contrafactuales.

## Contenido del laboratorio

### 1. Explicabilidad en imágenes

Se utiliza una `ResNet-18` con pesos preentrenados en ImageNet, configurada en modo evaluación. Las imágenes se redimensionan a 256 píxeles, se recortan al centro a `224 x 224` y se normalizan con las estadísticas de ImageNet.

Se analizan tres imágenes:

- **Imagen (a):** Labrador amarillo, con un objeto claro y centrado.
- **Imagen (b):** escena urbana con personas y un bus.
- **Imagen (c):** mandril en primer plano sobre un fondo complejo.

Para cada imagen se muestran las cinco clases más probables. Las predicciones principales observadas fueron:

| Imagen | Clase principal | Probabilidad |
|---|---|---:|
| (a) | Labrador Retriever | 0.7066 |
| (b) | minibus | 0.8284 |
| (c) | guenon | 0.5760 |

En la imagen (c) también se visualiza un *bounding box* en el recorte `224 x 224`, con coordenadas `(xmin, ymin, ancho, alto) = (50, 20, 130, 180)`.

#### Grad-CAM desde cero

Grad-CAM se implementa registrando *hooks* en `layer4` para capturar:

1. Las activaciones convolucionales.
2. Los gradientes del logit de la clase objetivo.
3. Los pesos de importancia de cada canal mediante el promedio espacial de los gradientes.
4. El mapa de activación final mediante una suma ponderada, `ReLU`, interpolación a `224 x 224` y normalización a `[0, 1]`.

Se generan mapas para la clase principal de cada imagen y para las dos clases principales de la imagen (b), con el fin de observar cómo cambia la atención espacial según la clase consultada.

#### CAM y verificación matemática

También se calcula CAM directamente a partir de los pesos de la capa completamente conectada y las activaciones de la red. Se verifica que, en esta arquitectura:

- El promedio espacial del mapa CAM más el sesgo reproduce el logit del modelo.
- El mapa CAM coincide con Grad-CAM sin `ReLU` ni normalización, salvo errores numéricos de punto flotante.
- Después del mismo posprocesamiento, ambos mapas producen prácticamente el mismo mapa de calor.

Los errores máximos observados fueron del orden de `10^-6` para la descomposición del logit y de `10^-7` para la comparación de mapas normalizados.

#### Análisis de atajos y causalidad

Para la imagen (c), solo el **34.39 %** de la masa del mapa Grad-CAM quedó dentro del *bounding box*. Además, al reemplazar el fondo por un valor gris neutro, la probabilidad de la clase `guenon` disminuyó de `0.5760` a `0.0002`, una caída relativa de **99.97 %**. Esto muestra que una explicación visual puede concentrarse parcialmente en el objeto, mientras que la predicción todavía depende fuertemente del contexto.

### 2. Explicabilidad tabular para riesgo crediticio

Se descarga el conjunto **German Credit** mediante `fetch_openml`. Las variables categóricas se codifican con `OrdinalEncoder` y se entrena un `GradientBoostingClassifier` con:

- División estratificada: 80 % entrenamiento y 20 % prueba.
- Semilla aleatoria: `42`.
- `200` estimadores.
- Profundidad máxima: `3`.
- Fondo SHAP: `100` instancias del conjunto de entrenamiento.

El modelo obtuvo:

| Métrica | Resultado |
|---|---:|
| Exactitud en prueba | 0.7750 |
| AUC en prueba | 0.7650 |

#### TreeSHAP e importancia de variables

Se utiliza `TreeSHAP` en el espacio de *log-odds* para:

- Verificar el axioma de eficiencia.
- Comprobar que el valor base coincide con la predicción promedio del fondo.
- Calcular la importancia global como el promedio de `|phi|`.
- Comparar la importancia SHAP con la importancia por permutación.
- Generar explicaciones locales (*waterfall*) y una explicación global (*beeswarm*).

La verificación de eficiencia presentó un error máximo de `9.59e-08`. El valor base fue `1.334498`, igual al promedio de *log-odds* del fondo. Las cinco variables principales según SHAP fueron:

1. `checking_status`
2. `credit_amount`
3. `purpose`
4. `duration`
5. `savings_status`

La correlación de Spearman entre los rankings de SHAP y permutación fue `0.7023`.

#### Variables redundantes

Se duplica la variable numérica más importante (`credit_amount`) agregando ruido gaussiano pequeño y se comparan dos configuraciones:

- **Intervencional:** utiliza el conjunto de fondo y rompe las dependencias entre variables.
- **Tree Path Dependent:** sigue las rutas de los árboles y aproxima una distribución condicional.

Este experimento muestra que la distribución de mérito entre variables correlacionadas depende del supuesto utilizado por el explicador.

#### KernelSHAP frente a TreeSHAP

Se comparan ambos explicadores sobre 10 instancias de prueba y fondos de 10, 50 y 100 observaciones:

| Fondo | Tiempo TreeSHAP | Tiempo KernelSHAP | MAE |
|---:|---:|---:|---:|
| 10 | 0.0139 s | 0.8668 s | 3.3860e-02 |
| 50 | 0.0228 s | 2.4206 s | 3.9121e-02 |
| 100 | 0.0234 s | 5.7245 s | 3.1422e-02 |

TreeSHAP es considerablemente más rápido para este modelo, mientras que KernelSHAP proporciona una aproximación model-agnostic con un costo computacional mayor.

#### Fórmula de Shapley desde cero

Para seis variables numéricas se enumeran las `2^6 = 64` coaliciones posibles y se calculan manualmente las contribuciones marginales. La implementación se valida contra TreeSHAP:

- Error absoluto máximo frente a TreeSHAP: `2.1303e-08`.
- Error máximo del axioma de eficiencia: `7.7716e-16`.

También se analiza la no linealidad al transformar *log-odds* a probabilidad. La suma de efectos individuales en probabilidad no coincide exactamente con el efecto total, porque la aditividad de SHAP se garantiza en el espacio de salida explicado, no después de aplicar la sigmoide.

El costo de enumerar todas las coaliciones crece exponencialmente:

- 20 variables: `1,048,576` evaluaciones, aproximadamente `45.25` minutos en la estimación del cuaderno.
- 60 variables: aproximadamente `3.65e+04` años suponiendo `1` microsegundo por evaluación.

### 3. Explicaciones locales con LIME

Se implementa LIME tabular sin utilizar la implementación oficial:

- Se generan `5,000` perturbaciones binarias.
- Se ponderan las muestras con un kernel exponencial y `sigma_rel = 0.75`.
- Se ajusta una regresión Ridge ponderada con regularización `lambda = 1.0`.
- El intercepto no se penaliza.
- Se seleccionan las cinco variables con mayor coeficiente absoluto.

Se explican tres clientes:

- Cliente A: probabilidad alta, `0.8664`.
- Cliente B: cliente rechazado, probabilidad `0.0871`.
- Cliente C: caso de frontera, probabilidad `0.4663`.

La comparación entre LIME y SHAP produjo correlaciones de Spearman de `0.8211`, `0.9564` y `0.8421` para los clientes A, B y C, respectivamente. En los tres casos coincidieron las tres variables principales.

La estabilidad de LIME se estudia con `N` igual a 500, 2,000 y 5,000, anchos de kernel de `0.5`, `0.75` y `1.5`, y 20 corridas por configuración. Se utilizan el índice de Jaccard de las cinco variables principales y el tamaño efectivo de muestra como métricas de estabilidad.

#### Explicaciones contrafactuales

Se generan 20,000 candidatos modificando como máximo tres variables accionables:

`duration`, `credit_amount`, `installment_commitment`, `checking_status` y `savings_status`.

Los candidatos se filtran por una probabilidad de aprobación de al menos `0.5`, se ordenan por una distancia normalizada y se seleccionan alternativas diversas. En la ejecución registrada:

- Fracción de candidatos válidos: `0.0550 %`.
- Distancia promedio: `6.2965`.
- Número promedio de cambios: `2.8182`.

### 4. Validación de explicaciones visuales

La calidad de Grad-CAM se evalúa con:

- **Deletion:** se eliminan progresivamente las regiones más importantes. Un `AUC_del` menor indica una explicación más fiel.
- **Insertion:** se revelan progresivamente las regiones más importantes desde una imagen desenfocada. Un `AUC_ins` mayor indica mejor desempeño.
- **Baseline aleatorio:** se comparan los resultados con 10 órdenes aleatorios de eliminación o inserción.

También se implementa el *sanity check* de Adebayo et al. (2018), reinicializando acumulativamente `fc`, `layer4`, `layer3`, `layer2`, `layer1` y `conv1`. La correlación de Spearman con el mapa original permite comprobar que la explicación depende de los parámetros aprendidos y no solo de la arquitectura.

Finalmente, se compara el uso de parches grises frente a un reemplazo desenfocado en Deletion para la imagen (a):

| Referencia de oclusión | AUC Deletion |
|---|---:|
| Parche gris | 0.0116 |
| Fondo desenfocado | 0.0686 |

El resultado evidencia que el tipo de referencia modifica la métrica: los parches grises pueden introducir bordes artificiales fuera de la distribución de entrenamiento.

## Librerías utilizadas

- [PyTorch](https://pytorch.org/) y [Torchvision](https://pytorch.org/vision/stable/)
- [Pillow](https://python-pillow.org/)
- [NumPy](https://numpy.org/) y [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [SciPy](https://scipy.org/)
- [SHAP](https://shap.readthedocs.io/)
- Jupyter

## Fuentes de datos e imágenes

- [Imagen Labrador - TensorFlow](https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg)
- [Imagen de bus - Ultralytics YOLOv5](https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg)
- [Imagen de mandril - OpenCV](https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg)
- [Etiquetas simples de ImageNet](https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json)
- Conjunto [German Credit (`credit-g`)](https://www.openml.org/search?type=data&status=active&id=31) mediante OpenML.

## Archivo principal

- [`Lab 9.ipynb`](./Lab%209.ipynb): implementación completa, visualizaciones, resultados y análisis.