# Evaluación Práctica 1
> **Fecha de entrega:** Viernes 13 de junio de 2025, 22:00
>
> **Nombre estudiante 1:** Fernando Canales Pérez
>

---

## Instrucciones Importantes
- Complete todas las secciones marcadas con `# TODO`
- No borre las salidas de las celdas una vez ejecutadas
- Este notebook debe ejecutarse de inicio a fin sin errores
- Incluya comentarios explicativos en su código
- Justifique todas sus decisiones analíticas
## Instalación de Dependencias
**Ejecute esta celda primero para instalar todas las librerías necesarias**
# Instalación de dependencias
!pip install pandas numpy matplotlib seaborn scikit-learn umap-learn
## Importación de Librerías
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP

import random
import numpy as np

# Para reproducibilidad
RNG_SEED = 42

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
---

## Pregunta 2: Análisis de columnas [1.5 pts.]

**Requisitos:**
- Listar cada columna con su tipo de dato
- Describir qué representa cada columna
- Incluir al menos 3 gráficos con análisis no trivial
- Describir conclusiones de cada gráfico
df.columns
Respuesta
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Columnas categóricas: {len(cat_cols)}")
print(df.select_dtypes(include='object').columns)
print(f"Columnas numéricas: {len(num_cols)}")
print( df.select_dtypes(include=['int64', 'float64']).columns)
**1. Análisis EDA - Primeros gráficos de histograma y boxplots para Variables Numéricas**

Conclusiones a partir de histogramas y boxplots
1. Distribuciones sesgadas hacia la derecha
* Muchas variables numericas como PrecioVenta, AreaLote, AreaHabitable, AreaGarage, etc presentan alta concentracion de valores bajos y cola larga hacia la dreceha.
*  Esto suguiere que la mayoria de las propiedades tienen valores moderados, pero hay algunas con valores mucho mas altos que el promedio.

2. Boxplots con bigotes derechos cargados de puntos
* Se observa una gran cantidad de puntos fuera del bigote, lo cual siguiere presencia de outliers altos.
* Aun que algunos podrían ser errores, en este contexto inmobiliario es más probable que sean propiedades grandes o de lujos.


# Seleccionar columnas numéricas
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Configurar la cantidad de subplots (cada columna tendrá histograma y boxplot)
n_cols = 4
n_rows = int(np.ceil(len(num_cols) * 2 / n_cols))  # 2 gráficos por columna

# Crear figura y ejes
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    # Histograma
    sns.histplot(df[col].dropna(), kde=True, ax=axes[2*i])
    axes[2*i].set_title(f'Histograma: {col}')

    # Boxplot
    sns.boxplot(x=df[col].dropna(), ax=axes[2*i + 1])
    axes[2*i + 1].set_title(f'Boxplot: {col}')

# Eliminar ejes vacíos
for j in range(2*len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
## Pregunta 3: Tratamiento de valores nulos [0.5 pts.]

**Nota:** Si su dataset no tiene valores nulos, esta pregunta no tendrá puntaje asignado.
Respuesta
import pandas as pd
import matplotlib.pyplot as plt

# Calcular proporción de valores nulos
null_ratios = df.isnull().mean()

# Filtrar columnas que tengan al menos un nulo
null_ratios = null_ratios[null_ratios > 0]

# Orden descendente
null_ratios = null_ratios.sort_values(ascending=False)

# Umbral para eliminación
umbral = 0.4
cols_a_eliminar = null_ratios[null_ratios > umbral]
cols_a_conservar = null_ratios[null_ratios <= umbral]

# Graficar
plt.figure(figsize=(14, 8))

# Barras por encima del umbral (en rojo)
plt.bar(cols_a_eliminar.index, cols_a_eliminar.values, label='Eliminar (nulos > 40%)', color='salmon')

# Barras por debajo o igual al umbral (en azul)
plt.bar(cols_a_conservar.index, cols_a_conservar.values, label='Conservar', color='skyblue')

# Líneas y etiquetas
plt.axhline(y=umbral, color='gray', linestyle='--', label=f'Umbral ({umbral*100:.0f}%)')
plt.xticks(rotation=90)
plt.ylabel('Proporción de valores nulos')
plt.title('Proporción de valores nulos por columna')
plt.legend()
plt.tight_layout()
plt.show()

### 🔧 Estrategia eliminacion de columnas por valores nulos:
Si una columna tiene más del 40-50% de valores nulos, se considera candidata a eliminación, ya que imputarla conlleva mucho sesgo o ruido. 
cols_a_eliminar
tipos_cols_conservar = df[cols_a_conservar.index].dtypes
print('✅ Tipos de columnas con valores nulos a inputar:')
print(tipos_cols_conservar)


✅ Estrategia de imputación y justificación por columna:
| Columna                        | Tipo    | Estrategia de imputación | Justificación                                                      |
| ------------------------------ | ------- | ------------------------ | ------------------------------------------------------------------ |
| `FrenteLote`                   | float64 | Mediana                  | Es una medida física continua → mediana evita sesgo por outliers.  |
| `TipoGarage`                   | object  | 'SinGarage'              | Si es nulo, indica que no hay garage.                              |
| `AnioGarage`                   | float64 | 0                        | Si no hay garage, no hay año de construcción.                      |
| `TerminacionGarage`            | object  | 'SinGarage'              | Ausencia → no aplica terminación.                                  |
| `CalidadGarage`                | object  | 'SinGarage'              | No aplica si no hay garage.                                        |
| `CondicionGarage`              | object  | 'SinGarage'              | Idem anterior.                                                     |
| `ExposicionSotano`             | object  | 'SinSotano'              | Si no hay exposición, probablemente no hay sótano.                 |
| `TipoFinSotano2`               | object  | 'SinTerminacion'         | Algunos sótanos solo tienen un acabado → segundo nulo = no existe. |
| `CalidadSotano`                | object  | 'SinSotano'              | Si es nulo, no hay sótano o no terminado.                          |
| `CondicionSotano`              | object  | 'SinSotano'              | Igual que anterior.                                                |
| `TipoFinSotano1`               | object  | 'SinTerminacion'         | Si no tiene terminación, nulo indica falta de acabado.             |
| `AreaRevestimientoMamposteria` | float64 | 0                        | Si no hay revestimiento, entonces área = 0.                        |
| `SistemaElectrico`             | object  | moda     | Usar moda (valor más frecuente).        |

cat_cols = df.select_dtypes(include=['object', 'category']).columns
num_cols = df.select_dtypes(exclude=['object', 'category']).columns


import matplotlib.pyplot as plt
import seaborn as sns

# Lista de columnas categóricas
cat_cols = [
    'Zonificacion', 'TipoCalle', 'FormaLote', 'ContornoTerreno',
    'Servicios', 'ConfiguracionLote', 'PendienteTerreno', 'Barrio',
    'Condicion1', 'Condicion2', 'TipoEdificio', 'EstiloCasa', 'EstiloTecho',
    'MaterialTecho', 'Exterior1', 'Exterior2', 'CalidadExterior',
    'CondicionExterior', 'Cimentacion', 'CalidadSotano', 'CondicionSotano',
    'ExposicionSotano', 'TipoFinSotano1', 'TipoFinSotano2',
    'SistemaCalefaccion', 'CalidadCalefaccion', 'AireAcondicionadoCentral',
    'SistemaElectrico', 'CalidadCocina', 'Funcionamiento', 'TipoGarage',
    'TerminacionGarage', 'CalidadGarage', 'CondicionGarage',
    'EntradaPavimentada', 'TipoVenta', 'CondicionVenta',
]

# Parámetros de la figura
n_cols = 4
n_rows = -(-len(cat_cols) // n_cols)  # Techo de la división entera
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()

# Graficar cada variable
for i, col in enumerate(cat_cols):
    sns.countplot(data=df, x=col, ax=axes[i], order=df[col].value_counts().index)
    axes[i].set_title(f'Distribución de {col}')
    axes[i].tick_params(axis='x', rotation=45)

# Eliminar subplots vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

import pandas as pd
import scipy.stats as stats

cat_cols = [
    'Zonificacion', 'TipoCalle', 'FormaLote', 'ContornoTerreno',
    'Servicios', 'ConfiguracionLote', 'PendienteTerreno', 'Barrio',
    'Condicion1', 'Condicion2', 'TipoEdificio', 'EstiloCasa', 'EstiloTecho',
    'MaterialTecho', 'Exterior1', 'Exterior2', 'CalidadExterior',
    'CondicionExterior', 'Cimentacion', 'CalidadSotano', 'CondicionSotano',
    'ExposicionSotano', 'TipoFinSotano1', 'TipoFinSotano2',
    'SistemaCalefaccion', 'CalidadCalefaccion', 'AireAcondicionadoCentral',
    'SistemaElectrico', 'CalidadCocina', 'Funcionamiento', 'TipoGarage',
    'TerminacionGarage', 'CalidadGarage', 'CondicionGarage',
    'EntradaPavimentada', 'TipoVenta', 'CondicionVenta'
]

anova_results = {}
for col in cat_cols:
    groups = [group["PrecioVenta"].values for name, group in df[[col, "PrecioVenta"]].dropna().groupby(col)]
    if len(groups) > 1:
        f_stat, p_val = stats.f_oneway(*groups)
        anova_results[col] = p_val

# Convertir a DataFrame y ordenar
anova_df = pd.DataFrame.from_dict(anova_results, orient='index', columns=["p_value"])
anova_df = anova_df.sort_values("p_value")
print('Resultado del ANOVA: Las 10 variables categóricas más asociadas con el PrecioVenta son:')
print(anova_df.head(10))  # Mostrar las 10 variables categóricas más asociadas

El resultado del ANOVA te indica que estas 10 variables categóricas tienen una relación estadísticamente significativa con el PrecioVenta

| Variable             | Interpretación práctica                                                  |
| -------------------- | ------------------------------------------------------------------------ |
| `Barrio`             | La zona tiene gran influencia en el precio (esperado).                   |
| `CalidadExterior`    | Mejor acabado exterior → mayor precio.                                   |
| `CalidadSotano`      | Un sótano de buena calidad impacta mucho en el valor.                    |
| `CalidadCocina`      | Cocinas de mayor calidad → precios más altos.                            |
| `TerminacionGarage`  | Garajes bien terminados elevan el precio.                                |
| `Cimentacion`        | Tipos de cimentación más sólidos se asocian a viviendas de mejor precio. |
| `TipoGarage`         | Tener garaje o su tipo afecta el valor.                                  |
| `TipoFinSotano1`     | Acabado del sótano primario es relevante.                                |
| `CalidadCalefaccion` | El confort térmico incide en el precio.                                  |
| `ExposicionSotano`   | Buena ventilación/luz en el sótano lo hace más útil y valioso.           |

import seaborn as sns
import matplotlib.pyplot as plt

top_vars = [
    'Barrio', 'CalidadExterior', 'CalidadSotano', 'CalidadCocina',
    'TerminacionGarage', 'Cimentacion', 'TipoGarage', 'TipoFinSotano1',
    'CalidadCalefaccion', 'ExposicionSotano'
]

for col in top_vars:
    # Ordenar categorías según la media del PrecioVenta
    orden = df.groupby(col)['PrecioVenta'].mean().sort_values().index

    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x=col, y='PrecioVenta', order=orden)
    plt.xticks(rotation=45)
    plt.title(f"PrecioVenta según {col}")
    plt.tight_layout()
    plt.show()

cat_cols
### 🔧 Estrategia eliminacion de columnas por baja correlacion con la Variable Objetivo (PrecioVenta):

---
# Fase 2: Ingeniería de Features (1.5 pts.)
cat_cols = df.select_dtypes(include=['object', 'category']).columns
cat_cols
# Imputación numérica
df['FrenteLote'] = df['FrenteLote'].fillna(df['FrenteLote'].median())
df['AnioGarage'] = df['AnioGarage'].fillna(0)
df['AreaRevestimientoMamposteria'] = df['AreaRevestimientoMamposteria'].fillna(0)

# Imputación categórica
df['TipoGarage'] = df['TipoGarage'].fillna('SinGarage')
df['TerminacionGarage'] = df['TerminacionGarage'].fillna('SinGarage')
df['CalidadGarage'] = df['CalidadGarage'].fillna('SinGarage')
df['CondicionGarage'] = df['CondicionGarage'].fillna('SinGarage')

df['ExposicionSotano'] = df['ExposicionSotano'].fillna('SinSotano')
df['TipoFinSotano1'] = df['TipoFinSotano1'].fillna('SinTerminacion')
df['TipoFinSotano2'] = df['TipoFinSotano2'].fillna('SinTerminacion')
df['CalidadSotano'] = df['CalidadSotano'].fillna('SinSotano')
df['CondicionSotano'] = df['CondicionSotano'].fillna('SinSotano')

# Para sistema eléctrico, usa moda (valor más frecuente)
df['SistemaElectrico'] = df['SistemaElectrico'].fillna(df['SistemaElectrico'].mode()[0])

print(df)
# Eliminar del DataFrame
df = df.drop(columns=cols_a_eliminar.index)
print(df.isnull().sum())
df.columns
import pandas as pd
import numpy as np

# 1. Seleccionar columnas numéricas (automáticamente excluye object/categorical)
df_numericas = df.select_dtypes(include=[np.number])

# 2. Calcular la correlación con PrecioVenta
correlaciones = df_numericas.corr()['PrecioVenta'].drop('PrecioVenta')

# 3. Definir umbral de baja correlación
umbral = 0.05
baja_corr = correlaciones[correlaciones.abs() < umbral]
alta_corr = correlaciones[correlaciones.abs() >= umbral]

# 4. Imprimir resultados
print("📉 Columnas con baja correlación con PrecioVenta (|corr| < {:.2f}):\n".format(umbral))
print(baja_corr.sort_values())

print("\n✅ Columnas con correlación significativa (|corr| ≥ {:.2f}):\n".format(umbral))
print(alta_corr.sort_values(ascending=False))
# 5. Graficar correlaciones significativas
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
correlaciones[correlaciones.abs() >= umbral].sort_values(ascending=False).plot(kind='bar')
plt.title('Correlación con PrecioVenta (|corr| ≥ {:.2f})'.format(umbral))
plt.ylabel('Correlación')
plt.xlabel('Columnas')
plt.axhline(y=umbral, color='red', linestyle='--', label='Umbral de baja correlación')
plt.axhline(y=-umbral, color='red', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()
# 6. Guardar DataFrame limpio
# df.to_csv('train_limpio.csv', index=False)
baja_corr
## Pregunta 4: Selección de columnas [0.5 pts]
En base en el análisis exploratorio y la matriz de correlación, se decidió eliminar las siguientes columnas:
# Eliminar columnas con baja correlación
df = df.drop(columns=baja_corr.index)
# Verificar columnas restantes
print("✅ Columnas restantes tras eliminar baja correlación:")
print(df.columns)

import matplotlib.pyplot as plt
import seaborn as sns

# Calcular correlaciones con PrecioVenta
df_numericas = df.select_dtypes(include=[np.number])
correlaciones = df_numericas.corr()['PrecioVenta'].drop('PrecioVenta')

# Tomar las 6 con mayor correlación absoluta
top_corr = correlaciones.abs().sort_values(ascending=False).head(6).index

# Graficar scatter plots
plt.figure(figsize=(15, 10))
for i, col in enumerate(top_corr, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=df[col], y=df['PrecioVenta'], alpha=0.6)
    plt.title(f'{col} vs PrecioVenta')
    plt.xlabel(col)
    plt.ylabel('PrecioVenta')

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Seleccionamos solo columnas numéricas
df_numericas = df.select_dtypes(include=[np.number])

# Calculamos la matriz de correlaciones
matriz_correlacion = df_numericas.corr()

# Tamaño del gráfico
plt.figure(figsize=(16, 12))

# Heatmap con anotaciones
sns.heatmap(
    matriz_correlacion,
    annot=True,              # muestra valores
    fmt=".2f",               # dos decimales
    cmap='coolwarm',         # colores
    vmin=-1, vmax=1,         # escala de correlación
    linewidths=0.5,          # separación entre celdas
    square=True,             # celdas cuadradas
    cbar_kws={"shrink": 0.75} # tamaño de la barra de color
)

plt.title("Mapa de Calor de Correlación entre Variables Numéricas", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

df.shape
df.shape
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Preparación
X_raw = df[df.columns.difference(['PrecioVenta'])]
y = df['PrecioVenta']

cat_cols = X_raw.select_dtypes(include=['object', 'category']).columns
num_cols = X_raw.select_dtypes(exclude=['object', 'category']).columns

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_cat_encoded = pd.DataFrame(encoder.fit_transform(X_raw[cat_cols]), columns=cat_cols, index=X_raw.index)

X = pd.concat([X_raw[num_cols], X_cat_encoded], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔸 Discretiza PrecioVenta en 5 rangos con etiquetas formateadas usando guion bajo
bins = pd.qcut(df['PrecioVenta'], q=5)
bin_labels = [
    f"{int(interval.left):_} → {int(interval.right):_}"
    for interval in bins.cat.categories
]
df['PrecioVenta_binned'] = pd.qcut(df['PrecioVenta'], q=5, labels=bin_labels)

# Mostrar los rangos utilizados en consola
print("Rangos de PrecioVenta (formato con _):")
for label in bin_labels:
    print(label)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Varianza explicada
print("Varianza explicada:", pca.explained_variance_ratio_)

# 🎨 📈 Gráfico mejorado
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df['PrecioVenta_binned'],
    palette=sns.color_palette("Set2"),
    alpha=0.75,
    s=60,
    edgecolor='black'
)

plt.title("Distribución PCA por rango de PrecioVenta", fontsize=14, weight='bold')
plt.xlabel(f"Componente 1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)", fontsize=12)
plt.ylabel(f"Componente 2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)", fontsize=12)
plt.legend(title="Rangos PrecioVenta", title_fontsize=11, fontsize=9)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

### Interpretacion de la Varianza explicada: [0.15599702 0.06029182] 
* 21.6% es bajo → Significa que la mayor parte de la variación de tus datos NO se puede visualizar claramente solo con 2 componentes.
* No significa que el análisis esté mal → solo que el dataset es complejo o tiene mucha dispersión en más dimensiones.

Para ver más varianza podria usar más componentes (3D o más → o usar técnicas como t-SNE o UMAP si te interesa visualización pura).
### ✅ 1. Justificación de la técnica eleccion PCA para reducción de dimensionalidad
Como la tarea es predicción del precio de venta, una variable continua, no se tiene una etiqueta categórica. Sin embargo, podemos discretizar el precio de venta para visualizar patrones.
Elegí PCA (Análisis de Componentes Principales) por las siguientes razones:

* Es útil como primer paso exploratorio.
* Permite visualizar la varianza explicada por componente.
* Requiere pocos hiperparámetros y es rápido de computar.

Aunque es lineal, puede revelar separaciones básicas en los datos.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

columnas_top_correlacion = [
    'CalidadGeneral', 'AreaHabitable', 'CapacidadGarage', 'AreaGarage','AreaTotalSotano', 'AreaPrimerPiso',
    'BaniosCompletos', 'TotalHabitaciones', 'AnioConstruccion', 'AnioRemodelacion', 'AreaRevestimientoMamposteria'
]
df_top_corr = df[columnas_top_correlacion + ['PrecioVenta']].copy()

# 1. Preparar los datos
y = df_top_corr['PrecioVenta']
X = df_top_corr.drop(columns=['PrecioVenta'])

# 2. Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Discretiza el target para graficar
df['PrecioVenta_binned'] = pd.qcut(df['PrecioVenta'], q=3, labels=["Bajo", "Medio", "Alto"])

# 4. Configuraciones de componentes
n_features = X.shape[1]
n_components_list = [n for n in [2, 5, 10, 20] if n <= n_features]

# 5. Gráficos de dispersión PCA
fig, axes = plt.subplots(1, len(n_components_list), figsize=(5 * len(n_components_list), 5))
if len(n_components_list) == 1:
    axes = [axes]

for i, n in enumerate(n_components_list):
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X_scaled)

    ax = axes[i]
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['PrecioVenta_binned'], palette="viridis", ax=ax)
    ax.set_title(f"PCA con {n} componentes\n(varianza PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca.explained_variance_ratio_[1]*100:.1f}%)\nTotal varianza explicada: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="PrecioVenta")

plt.tight_layout()
plt.suptitle("Comparación de PCA con distintas configuraciones", fontsize=16, y=1.02)
plt.show()

# ✅ Imprimir varianza explicada acumulada por cada configuración
print("Varianza explicada acumulada por configuración de PCA:")
for n in n_components_list:
    pca = PCA(n_components=n)
    pca.fit(X_scaled)
    explained_var = pca.explained_variance_ratio_.cumsum()
    print(f"- Con {n} componentes: {explained_var[-1]*100:.2f}% varianza explicada acumulada")

# ✅ Gráfico adicional: Varianza explicada acumulada
fig, ax = plt.subplots(figsize=(8, 5))

for n in n_components_list:
    pca = PCA(n_components=n)
    pca.fit(X_scaled)
    explained_var = pca.explained_variance_ratio_.cumsum()
    ax.plot(range(1, n+1), explained_var, marker='o', label=f'{n} componentes')

ax.set_xlabel("Número de componentes")
ax.set_ylabel("Varianza explicada acumulada")
ax.set_title("Comparación de varianza explicada acumulada por configuración de PCA")
ax.legend(title="Configuración")
ax.grid(True)
plt.show()

Interpretación de la varianza explicada acumulada en PCA
Con 2 componentes (62.33%):
Los dos primeros componentes principales capturan aproximadamente el 62% de la variabilidad total en los datos. Esto significa que con solo dos dimensiones ya tienes una representación bastante compacta, pero aún se pierde cerca del 38% de la información original. Puede ser útil para visualizaciones rápidas, pero para un análisis más detallado probablemente sea insuficiente.

Con 5 componentes (86.67%):
Al usar cinco componentes, retienes casi el 87% de la varianza original, lo que indica que la mayoría de la información relevante del dataset está concentrada en estos cinco ejes principales. Esta configuración es un buen compromiso entre reducción de dimensionalidad y preservación de información, ideal para análisis o modelos que buscan simplificar pero conservar calidad.

Con 10 componentes (99.05%):
Con diez componentes, prácticamente estás reteniendo toda la variabilidad (más del 99%) de los datos originales. Esto significa que casi no pierdes información importante, pero la reducción de dimensionalidad es menos significativa. En casos donde la pérdida mínima de información es crítica, esta es la mejor opción, aunque puede ser más costosa computacionalmente.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

columnas_top_correlacion = [
    'CalidadGeneral', 'AreaHabitable', 'CapacidadGarage', 'AreaGarage','AreaTotalSotano', 'AreaPrimerPiso',
    'BaniosCompletos', 'TotalHabitaciones', 'AnioConstruccion', 'AnioRemodelacion', 'AreaRevestimientoMamposteria'
]
df_top_corr = df[columnas_top_correlacion + ['PrecioVenta']].copy()

# 1. Preparar los datos
y = df_top_corr['PrecioVenta']
X = df_top_corr.drop(columns=['PrecioVenta'])

# 3. Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Discretiza después de preparar X
df['PrecioVenta_binned'] = pd.qcut(df['PrecioVenta'], q=3, labels=["Bajo", "Medio", "Alto"])

# ✅ Asegúrate de no pedir más componentes que columnas
n_features = X.shape[1]
n_components_list = [n for n in [2, 5, 10, 20] if n <= n_features]

# 5. Comparación de PCA con distintas cantidades de componentes
fig, axes = plt.subplots(1, len(n_components_list), figsize=(5 * len(n_components_list), 5))
if len(n_components_list) == 1:
    axes = [axes]  # para que sea iterable

for i, n in enumerate(n_components_list):
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X_scaled)

    ax = axes[i]
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['PrecioVenta_binned'], palette="viridis", ax=ax)
    ax.set_title(f"PCA con {n} componentes\nPC1: {pca.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="PrecioVenta")

plt.tight_layout()
plt.suptitle("Comparación de PCA con distintas configuraciones", fontsize=16, y=1.02)
plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP

# 1. Separa columnas a conservar
X_raw = df[df.columns.difference(['PrecioVenta'])]
y = df['PrecioVenta']

# 2. Codifica columnas categóricas
cat_cols = X_raw.select_dtypes(include='object').columns
num_cols = X_raw.select_dtypes(exclude='object').columns

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_cat_encoded = pd.DataFrame(encoder.fit_transform(X_raw[cat_cols]), columns=cat_cols, index=X_raw.index)

# 3. Combina numéricas y categóricas codificadas
X = pd.concat([X_raw[num_cols], X_cat_encoded], axis=1)

# 4. Escala los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Aplicar UMAP
umap = UMAP(n_components=2, random_state=42)
X_umap = umap.fit_transform(X_scaled)

# 6. Graficar
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette='viridis', alpha=0.7)
plt.title('Visualización con UMAP coloreada por PrecioVenta')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(title='PrecioVenta', loc='upper right')
plt.tight_layout()
plt.show()

top_corr.get_indexer_non_unique
## Pregunta 5: Análisis por visualización de datos [1.0 pts.]

**Técnicas disponibles:** PCA, t-SNE, UMAP
Respuesta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

columnas_top_correlacion = [
    'CalidadGeneral', 'AreaHabitable', 'CapacidadGarage', 'AreaGarage','AreaTotalSotano', 'AreaPrimerPiso'
]
df_top_corr = df[columnas_top_correlacion + ['PrecioVenta']].copy()
# 1. Preparar los datos
y = df_top_corr['PrecioVenta']
X = df_top_corr.drop(columns=['PrecioVenta'])

# 2. One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# 3. Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Reducción de dimensionalidad
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

umap = UMAP(n_components=2, random_state=42)
X_umap = umap.fit_transform(X_scaled)

# 5. Visualización
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# PCA
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', ax=axes[0], alpha=0.7)
axes[0].set_title('PCA')

# t-SNE
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='viridis', ax=axes[1], alpha=0.7)
axes[1].set_title('t-SNE')

# UMAP
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette='viridis', ax=axes[2], alpha=0.7)
axes[2].set_title('UMAP')

# Ajustes generales
for ax in axes:
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')
    ax.legend([],[], frameon=False)  # Oculta la leyenda repetida

plt.suptitle('Comparación PCA vs t-SNE vs UMAP según PrecioVenta', fontsize=14)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
'''
# 1. Preparar los datos
y = df['PrecioVenta']
X = df.drop(columns=['PrecioVenta'])

# 2. One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# 3. Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
'''
# 4. Reducción de dimensionalidad
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

umap = UMAP(n_components=2, random_state=42)
X_umap = umap.fit_transform(X_scaled)

# 5. Visualización
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# PCA
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', ax=axes[0], alpha=0.7)
axes[0].set_title('PCA')

# t-SNE
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='viridis', ax=axes[1], alpha=0.7)
axes[1].set_title('t-SNE')

# UMAP
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette='viridis', ax=axes[2], alpha=0.7)
axes[2].set_title('UMAP')

# Ajustes generales
for ax in axes:
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')
    ax.legend([],[], frameon=False)  # Oculta la leyenda repetida

plt.suptitle('Comparación PCA vs t-SNE vs UMAP según PrecioVenta', fontsize=14)
plt.tight_layout()
plt.show()

---
# Fase 3: Modelado y Evaluación (4 pts.)
## Pregunta 6: Partición de datos [0.5 pts.]
Respuesta
# Completar
df.shape
#### ✅ Recomendación general

**📌 Porcentaje de división**

* Entrenamiento: 80%
* Evaluación (test): 20%

Esta proporción es una práctica estándar cuando tienes un tamaño de muestra razonable como 1460 filas.

**Justificación:**

* 80% permite entrenar con suficiente información (1168 muestras aprox).
* 20% es representativo para una evaluación confiable (292 muestras aprox).
* Minimiza el overfitting y garantiza una evaluación justa.
from sklearn.model_selection import train_test_split

# Suponiendo que ya separaste X e y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       # 20% para test
    random_state=RNG_SEED      # para reproducibilidad
)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
## Pregunta 7: Predicción con modelos de ML [2 pts.]

**Modelos disponibles:** KNN, SVM, Árboles de Decisión, Random Forest, Gradient Boosting, MLP
Respuesta
# ✅ Entrenamiento de modelos y evaluación

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inicializar modelos
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Entrenamiento
rf.fit(X_train_scaled, y_train)
gb.fit(X_train_scaled, y_train)

# Predicciones
y_pred_rf = rf.predict(X_test_scaled)
y_pred_gb = gb.predict(X_test_scaled)

# Evaluación
def evaluar_modelo(nombre, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5  # sqrt for RMSE
    r2 = r2_score(y_true, y_pred)
    print(f"📊 Resultados de {nombre}:")
    print(f"MAE :  {mae:.2f}")
    print(f"RMSE:  {rmse:.2f}")
    print(f"R²   :  {r2:.2f}\n")

evaluar_modelo("Random Forest", y_test, y_pred_rf)
evaluar_modelo("Gradient Boosting", y_test, y_pred_gb)

Resumen general
* Random Forest y Gradient Boosting son modelos sólidos y funcionan muy bien en tu problema, con alta capacidad predictiva y bajo error.

* Gradient Boosting tiene una ligera ventaja en MAE, lo que puede hacerlo preferible si buscas minimizar el error absoluto.
# ✅ Gráfico de Predicción vs Real

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred_gb, alpha=0.6, label='Gradient Boosting')
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.6, label='Random Forest')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('PrecioVenta real')
plt.ylabel('PrecioVenta predicho')
plt.title('Predicción vs Real')
plt.legend()
plt.tight_layout()
plt.show()

## Pregunta 8: Sobreentrenamiento [1.5 pts.]
Respuesta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Rango de profundidad
depths = range(1, 21)
r2_train_rf = []
r2_test_rf = []
rmse_train_rf = []
rmse_test_rf = []

for depth in depths:
    model_rf = RandomForestRegressor(max_depth=depth, n_estimators=100, random_state=42)
    model_rf.fit(X_train_scaled, y_train)

    y_train_pred_rf = model_rf.predict(X_train_scaled)
    y_test_pred_rf = model_rf.predict(X_test_scaled)

    r2_train_rf.append(r2_score(y_train, y_train_pred_rf))
    r2_test_rf.append(r2_score(y_test, y_test_pred_rf))

    rmse_train_rf.append(np.sqrt(mean_squared_error(y_train, y_train_pred_rf)))
    rmse_test_rf.append(np.sqrt(mean_squared_error(y_test, y_test_pred_rf)))

# Subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico R²
axes[0].plot(depths, r2_train_rf, marker='o', label='R² entrenamiento')
axes[0].plot(depths, r2_test_rf, marker='o', label='R² test')
axes[0].set_xlabel('Profundidad máxima (max_depth)')
axes[0].set_ylabel('R²')
axes[0].set_title('Overfitting: R² vs Profundidad')
axes[0].legend()
axes[0].grid(True)

# Gráfico RMSE
axes[1].plot(depths, rmse_train_rf, marker='o', label='RMSE entrenamiento')
axes[1].plot(depths, rmse_test_rf, marker='o', label='RMSE test')
axes[1].set_xlabel('Profundidad máxima (max_depth)')
axes[1].set_ylabel('RMSE')
axes[1].set_title('Error: RMSE vs Profundidad')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.suptitle('Evaluación de Random Forest según profundidad de árboles', fontsize=16, y=1.05)
plt.show()

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Rango de profundidad
depths = range(1, 21)
r2_train_rf = []
r2_test_rf = []
rmse_train_rf = []
rmse_test_rf = []

for depth in depths:
    model_rf = GradientBoostingRegressor(max_depth=depth, n_estimators=100, random_state=42)
    model_rf.fit(X_train_scaled, y_train)

    y_train_pred_rf = model_rf.predict(X_train_scaled)
    y_test_pred_rf = model_rf.predict(X_test_scaled)

    r2_train_rf.append(r2_score(y_train, y_train_pred_rf))
    r2_test_rf.append(r2_score(y_test, y_test_pred_rf))

    rmse_train_rf.append(np.sqrt(mean_squared_error(y_train, y_train_pred_rf)))
    rmse_test_rf.append(np.sqrt(mean_squared_error(y_test, y_test_pred_rf)))

# Subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico R²
axes[0].plot(depths, r2_train_rf, marker='o', label='R² entrenamiento')
axes[0].plot(depths, r2_test_rf, marker='o', label='R² test')
axes[0].set_xlabel('Profundidad máxima (max_depth)')
axes[0].set_ylabel('R²')
axes[0].set_title('Overfitting: R² vs Profundidad')
axes[0].legend()
axes[0].grid(True)

# Gráfico RMSE
axes[1].plot(depths, rmse_train_rf, marker='o', label='RMSE entrenamiento')
axes[1].plot(depths, rmse_test_rf, marker='o', label='RMSE test')
axes[1].set_xlabel('Profundidad máxima (max_depth)')
axes[1].set_ylabel('RMSE')
axes[1].set_title('Error: RMSE vs Profundidad')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.suptitle('Evaluación de GradientBoostingRegressor según profundidad de árboles', fontsize=16, y=1.05)
plt.show()
