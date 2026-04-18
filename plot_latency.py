import matplotlib.pyplot as plt
import numpy as np

# 1. Definición de la Paleta de Colores (Estilo Slide 3)
COLOR_FONDO = '#5E0B9E'       # Violeta profundo corporativo
COLOR_TEXTO = '#FFFFFF'       # Blanco puro
COLOR_FLOAT32 = '#B868F8'     # Violeta claro (Línea base)
COLOR_BFLOAT16 = '#FF4B4B'    # Salmón/Naranja (Alerta/Penalización)
COLOR_HIGHLIGHT = '#FFF000'   # Amarillo neón (Anotaciones)

# 2. Datos del Reporte de Análisis
etiquetas = ['Float32\n(Precisión Estándar)', 'BFloat16\n(Mixed Precision)']
latencia_ms = [628, 700]
colores_barras = [COLOR_FLOAT32, COLOR_BFLOAT16]

# 3. Configuración del Entorno Matplotlib
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(COLOR_FONDO)
ax.set_facecolor(COLOR_FONDO)

# 4. Creación del Gráfico de Barras
barras = ax.bar(etiquetas, latencia_ms, color=colores_barras, width=0.5, edgecolor=COLOR_TEXTO, linewidth=1.5)

# 5. Personalización de Ejes y Tipografía
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(COLOR_TEXTO)
ax.spines['bottom'].set_color(COLOR_TEXTO)

ax.tick_params(axis='x', colors=COLOR_TEXTO, labelsize=12)
ax.tick_params(axis='y', colors=COLOR_TEXTO, labelsize=12)
ax.set_ylabel('Latencia por Paso (ms)', color=COLOR_TEXTO, fontsize=14, fontweight='bold')
ax.set_title('Impacto Arquitectónico TPU v6e: Float32 vs BFloat16', color=COLOR_TEXTO, fontsize=16, fontweight='bold', pad=20)

# 6. Anotaciones de Datos sobre las Barras
for bar in barras:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval - 40, f'{int(yval)} ms', 
            ha='center', va='top', color=COLOR_TEXTO, fontsize=14, fontweight='bold')

# 7. Anotación Crítica del "Delta" (+11.5%)
ax.annotate('+ 11.5% Overhead\n(Casting Cost)', 
            xy=(1, 700), 
            xytext=(1, 750),
            ha='center',
            color=COLOR_HIGHLIGHT,
            fontsize=13,
            fontweight='bold',
            arrowprops=dict(facecolor=COLOR_HIGHLIGHT, edgecolor=COLOR_HIGHLIGHT, shrink=0.05, width=2, headwidth=8))

# 8. Ajuste de límites para dejar espacio a las anotaciones
ax.set_ylim(0, 850)

# Renderizado final (Se recomienda exportar como .svg para la presentación)
plt.tight_layout()
# Guardamos respetando el color de fondo para la presentación visual
plt.savefig('latencia_precision.png', dpi=300, facecolor=COLOR_FONDO)
print('Graphic Generated Successfully.')
