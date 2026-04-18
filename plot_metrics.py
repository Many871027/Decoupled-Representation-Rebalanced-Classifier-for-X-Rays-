import matplotlib.pyplot as plt
import seaborn as sns

# Datos proporcionados extraídos de la tabla
epochs = [1, 3, 5, 7, 8, 9, 11, 13, 15, 17]
val_macro_f1 = [0.2210, 0.3140, 0.4850, 0.5890, 0.6500, 0.8355, 0.8120, 0.7840, 0.7410, 0.7105]
val_f1_covid = [0.0000, 0.0400, 0.3500, 0.4900, 0.5200, 0.7500, 0.7200, 0.6500, 0.5900, 0.5500]

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Trazado de líneas solicitadas
plt.plot(epochs, val_macro_f1, label='Val Macro-F1 (Rendimiento Global)', color='#1f77b4', linestyle='-', linewidth=2.5, marker='o')
plt.plot(epochs, val_f1_covid, label='Val F1 (COVID Minoritario)', color='#d62728', linestyle='--', linewidth=2.5, marker='s')

# Anotación Época 8: ReduceLROnPlateau
plt.axvline(x=8, color='gray', linestyle=':', alpha=0.7)
plt.text(8.2, 0.1, 'ReduceLROnPlateau\n(LR → 1e-4)', color='gray', fontsize=10, verticalalignment='bottom')

# Anotación Época 9: Vértice Óptimo
plt.scatter([9], [0.8355], color='gold', s=150, zorder=5, edgecolors='black')
plt.annotate('🏆 Vértice Óptimo\n(Guardado de Pesos)', xy=(9, 0.8355), xytext=(9.5, 0.95),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5), fontsize=11, fontweight='bold', color='darkgreen')

# Anotación Época 17: Early Stopping
plt.axvline(x=17, color='maroon', linestyle=':', alpha=0.7)
plt.text(16.5, 0.1, '🛑 Early Stopping\n(Restauración a Ep. 9)', color='maroon', fontsize=10, horizontalalignment='right', verticalalignment='bottom')

plt.title('Dinámica de Convergencia: Impacto de la Clase Minoritaria', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Época de Entrenamiento', fontsize=12, fontweight='bold')
plt.ylabel('Puntuación F1 (F1-Score)', fontsize=12, fontweight='bold')
plt.legend(loc='center right', fontsize=11)
plt.xticks(epochs)
plt.ylim(0, 1.05)

plt.tight_layout()
plt.savefig('convergencia_cola_larga.png', dpi=300)
print('Graphic Generated Successfully.')
