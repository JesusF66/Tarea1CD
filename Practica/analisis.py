import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Establecemos el working directory
os.chdir(".\\Practica\\Data\\2023-Data")

# Lectura de datos
df = pd.read_excel("data.xlsx", header=None)


# Analisis de datos faltantes

missing_count = df.isna().sum()
missing_percent = (missing_count / len(df)) * 100
missing_summary = pd.DataFrame(
    {"Valores faltantes": missing_count, "Porcentaje": missing_percent.round(2)}
)

print("\n--- Datos faltantes por columna ---")
print(missing_summary)


# Deteccion de outliers

# Para reemplazar los NA + espacio(s) por NA
df = df.replace("NA", "NaN")
df = df.replace("NA ", "NaN")
df = df.replace("NA  ", "NaN")

# Seleccionamos solamente los datos de 13CVPDB para cada columna
isotope_columns = [df[df.columns[i]].iloc[10:] for i in range(1, len(df.columns))]

# Metodo 1: Z-score
print("\nOutlier detection using Z-score method (|Z| > 3):")
zscore_outliers = {}

for i in range(len(isotope_columns)):
    # Eliminamos los NA para los calculos
    col = isotope_columns[i]
    data = pd.DataFrame(col.dropna(), dtype="float")

    # Calculamos los z-scores y seleccionamos los posibles outliers
    z_scores = np.abs(stats.zscore(data))
    outliers = data[z_scores > 3]
    outliers = pd.DataFrame(outliers.values).dropna()
    zscore_outliers[i] = len(outliers)

    # Imprimimos los posibles outliers (si hay), y las fechas a las que corresponden
    if len(outliers) > 0:
        print(f"{df[i].iloc[0]}: {len(outliers)} outliers")
        print(f"  Outlier values:\n{outliers}")
        print(f"  Outlier years: {df[0].iloc[outliers.index].values}")

# Metodo 2: IQR rango intercuantilico
print("\nOutlier detection using IQR method:")
iqr_outliers = {}

for i in range(len(isotope_columns)):
    # Eliminamos los NA para los calculos
    col = isotope_columns[i]
    data = pd.DataFrame(col.dropna(), dtype="float")

    # Calculamos el rango intercuartilico y seleccionamos los posibles outliers
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data < lower_bound) | (data > upper_bound)]
    outliers = pd.DataFrame(outliers.values).dropna()
    iqr_outliers[i] = len(outliers)

    # Imprimimos los posibles outliers (si hay), y las fechas a las que corresponden
    if len(outliers) > 0:
        print(f"{df[i].iloc[0]}: {len(outliers)} outliers")
        print(f"  Outlier values:\n {outliers.values}")
        print(f"  Outlier years: {df[0].iloc[outliers.index].values}")

# Visualizamos los outliers posibles
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Z-score
zscore_counts = [zscore_outliers[i] for i in range(len(df.columns) - 1)]
axes[0].bar([df[i + 1].iloc[0] for i in range(len(isotope_columns))], zscore_counts)
axes[0].set_title("Outliers detected by Z-score method (|Z| > 3)", fontweight="bold")
axes[0].set_ylabel("Number of outliers")
axes[0].tick_params(axis="x", rotation=45)

# IQR
iqr_counts = [iqr_outliers[i] for i in range(len(df.columns) - 1)]
axes[1].bar([df[i + 1].iloc[0] for i in range(len(isotope_columns))], iqr_counts)
axes[1].set_title("Outliers detected by IQR method", fontweight="bold")
axes[1].set_ylabel("Number of outliers")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# Inconsistencias o codificacion ambigua

# Revisamos si hay fechas duplicadas
duplicate_years = df[0].iloc[10:].duplicated()
if duplicate_years.any():
    print(f"Warning: {duplicate_years.sum()} duplicate years found!")
    print(df["Year CE"][duplicate_years].values)
else:
    print("No duplicate years found.")

# Imprimimos el rango de fechas disponibles para cada localizacion
print("\nYear range for each site:")

for i in range(len(isotope_columns)):
    # Remove NA values for calculation
    col = isotope_columns[i]
    data = pd.DataFrame(col.dropna(), dtype="float")

    if len(data) > 0:
        years_with_data = df[0].iloc[data.index]
        print(
            f"{df[i].iloc[0]}: {years_with_data.min()} - {years_with_data.max()} "
            + f"({len(data)} years of data)"
        )

# Imprimimos el numero de diferentes codigos de zonas
print(f"\nNumber of different codes: {len(df[1:].iloc[0].unique())}")

# 3. IMPUTACION DE DATOS

df_isotopes = pd.DataFrame(isotope_columns).T
df_isotopes = pd.DataFrame(df_isotopes, dtype="float")
df_isotopes = df_isotopes.set_axis(df.iloc[0][1:], axis=1)
df_isotopes = df_isotopes.set_index(df[0].iloc[10:])

# Reemplazo con media
df_mean_imput = df_isotopes.fillna(df_isotopes.mean())

# Reemplazo con interpolacion lineal
df_linear_imput = df_isotopes.interpolate(method="linear")

# Para guardarlos en archivos de Excel
df_mean_imput.to_excel("meanimputation.xlsx")
df_linear_imput.to_excel("linearimputation.xlsx")


# 4. Codificacion y escalamiento
