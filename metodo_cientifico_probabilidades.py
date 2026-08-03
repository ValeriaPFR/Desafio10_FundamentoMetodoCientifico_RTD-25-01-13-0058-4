import pandas as pd
import numpy as np

# Configuración de semilla para reproducibilidad
np.random.seed(42)

# Crear la población de 1000 usuarios
data = {
    'user_id': range(1, 1001),
    'pais': np.random.choice(['México', 'Colombia', 'Argentina', 'Chile'], 1000, p=[0.4, 0.3, 0.2, 0.1]),
    'tiene_premium': np.random.choice([True, False], 1000, p=[0.35, 0.65])
}
poblacion_df = pd.DataFrame(data)

# Asegurar que algunos usuarios de Chile tengan premium
idx_chile = poblacion_df[poblacion_df['pais'] == 'Chile'].index
poblacion_df.loc[np.random.choice(idx_chile, 10, replace=False), 'tiene_premium'] = True

# 1. Probabilidad Teórica (Poblacional)
total = len(poblacion_df)
P_A = len(poblacion_df[poblacion_df['pais'] == 'Chile']) / total
P_B = len(poblacion_df[poblacion_df['tiene_premium'] == True]) / total
P_A_int_B = len(poblacion_df[(poblacion_df['pais'] == 'Chile') & (poblacion_df['tiene_premium'] == True)]) / total
P_A_uni_B = P_A + P_B - P_A_int_B

# 2. Muestreo Aleatorio Simple (n=150)
muestra_simple = poblacion_df.sample(n=150, random_state=42)
P_B_simple = muestra_simple['tiene_premium'].mean()

# 3. Muestreo Estratificado (proporción 0.15)
muestra_estratificada = poblacion_df.groupby('pais', group_keys=False).apply(lambda x: x.sample(frac=0.15, random_state=42))
P_B_estratificada = muestra_estratificada['tiene_premium'].mean()

# --- IMPRESIÓN DE RESULTADOS EN TERMINAL ---
print("=" * 60)
print(" REPORTE DE ANÁLISIS PROBABILÍSTICO Y MUESTREO")
print("=" * 60)

print("\n--- 1. PROBABILIDAD TEÓRICA (POBLACIÓN TOTAL N = 1000) ---")
print(f"P(Chile) [A]: {P_A:.4f} ({P_A * 100:.2f}%)")
print(f"P(Premium) [B]: {P_B:.4f} ({P_B * 100:.2f}%)")
print(f"P(Chile ∩ Premium): {P_A_int_B:.4f} ({P_A_int_B * 100:.2f}%)")
print(f"P(Chile ∪ Premium): {P_A_uni_B:.4f} ({P_A_uni_B * 100:.2f}%)")

print("\n--- 2. MUESTREO ALEATORIO SIMPLE (n = 150) ---")
print(f"Estimación P(Premium) en muestra simple: {P_B_simple:.4f} ({P_B_simple * 100:.2f}%)")

print("\n--- 3. MUESTREO ESTRATIFICADO (frac = 0.15) ---")
print(f"Estimación P(Premium) en muestra estratificada: {P_B_estratificada:.4f} ({P_B_estratificada * 100:.2f}%)")

print("\n--- 4. ANÁLISIS COMPARATIVO ---")
diff_simple = abs(P_B_simple - P_B)
diff_estratificada = abs(P_B_estratificada - P_B)

print(f"Valor real poblacional P(B): {P_B:.4f}")
print(f"Error Muestreo Simple:         {diff_simple:.4f}")
print(f"Error Muestreo Estratificado:  {diff_estratificada:.4f}")
print("-" * 60)
if diff_estratificada < diff_simple:
    print("Conclusión: El muestreo estratificado presenta una mayor precisión")
    print("al reducir el error de estimación respecto al parámetro poblacional.")
else:
    print("Conclusión: En esta ejecución particular, el muestreo aleatorio simple")
    print("mostró una aproximación comparable o superior.")
print("=" * 60)