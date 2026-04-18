import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. Definición de Paleta Corporativa (Base Violeta)
COLOR_FONDO = '#5E0B9E'       
COLOR_CAJA = '#3A0368'        # Cajas un tono más oscuro para contraste interno
COLOR_TEXTO = '#FFFFFF'       
COLOR_HIGHLIGHT = '#FFF000'   

# Configuración del entorno base
fig, ax = plt.subplots(figsize=(14, 5.5))
fig.patch.set_facecolor(COLOR_FONDO)
ax.set_facecolor(COLOR_FONDO)
ax.axis('off')  # Ocultar todos los ejes

# Título de la infografía
plt.text(0.5, 0.95, 'RESULTADOS ESTRATÉGICOS: DeRC-X',
         ha='center', va='center', color=COLOR_TEXTO, fontsize=22, fontweight='bold', transform=ax.transAxes)

# Parámetros geométricos para 3 cajas (Columnas)
box_width = 0.28
box_height = 0.60
gap = (1.0 - 3 * box_width) / 4

# Contenido exacto solicitado
boxes_data = [
    {
        "title": "Validación Empírica", 
        "text": "Macro-F1: 0.8355\n\nfrente a\n\ndesbalance 11:1"
    },
    {
        "title": "Alta Precisión Diagnóstica", 
        "text": "COVID-19\nPrecisión: 0.79\n\n(4 de 5 alertas son\nclínicamente reales)"
    },
    {
        "title": "Eficiencia Computacional", 
        "text": "Arquitectura Custom CNN\n+\nResolución Nativa (512x512)\n\nen Float32"
    }
]

# Trazado e Inyección de Texto
for i, data in enumerate(boxes_data):
    # Coordenadas X ancladas dinámicamente
    x0 = gap + i * (box_width + gap)
    y0 = 0.15
    
    # Renderizado de Caja (Fondo oscuro, borde neón)
    box = patches.FancyBboxPatch(
        (x0, y0), box_width, box_height,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor=COLOR_CAJA, edgecolor=COLOR_HIGHLIGHT, lw=2, transform=ax.transAxes
    )
    ax.add_patch(box)
    
    # Centro geométrico de la caja para alineaciones
    center_x = x0 + box_width / 2
    
    # Renderizado del Título (Amarillo Neón)
    ax.text(center_x, y0 + box_height - 0.08, data['title'].upper(),
            ha='center', va='top', color=COLOR_HIGHLIGHT, fontsize=13, fontweight='900', transform=ax.transAxes)
            
    # Renderizado del Cuerpo (Blanco Puro)
    ax.text(center_x, y0 + box_height/2 - 0.05, data['text'],
            ha='center', va='center', color=COLOR_TEXTO, fontsize=15, fontweight='bold', transform=ax.transAxes)

plt.tight_layout()

# Guardado conservando el fondo violeta
plt.savefig('columnas_impacto.png', dpi=300, facecolor=COLOR_FONDO)
print('Graphic Generated Successfully.')
