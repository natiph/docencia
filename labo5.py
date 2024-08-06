# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:56:05 2024

@author: Natalia
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para convertir RGB a una longitud de onda aproximada
def rgb_to_wavelength(r, g, b):
    if g >= r and g >= b:
        green_intensity = g / 255.0  # Normalizar a 0-1
        wavelength = 500 + green_intensity * 70  # Mapea a 500-570 nm
        return wavelength
    elif r >= g and r >= b:
        yellow_intensity = g / 255.0  # Normalizar a 0-1
        wavelength = 570 + yellow_intensity * 20  # Mapea a 570-590 nm
        return wavelength
    elif b >= g and b >= r:
        blue_intensity = b / 255.0  # Normalizar a 0-1
        wavelength = 450 + blue_intensity * 45  # Mapea a 450-495 nm
        return wavelength
    else:
        return np.nan  # Asignar NaN para otros colores

# Leer la imagen
imagen = cv2.imread(r"C:\Users\Usuario\Downloads\hojas4.jpg")

# Convertir la imagen de BGR a RGB (OpenCV carga la imagen en BGR por defecto)
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Extraer los canales rojo, verde y azul
canal_rojo = imagen_rgb[:, :, 0]
canal_verde = imagen_rgb[:, :, 1]
canal_azul = imagen_rgb[:, :, 2]

# Crear una máscara para colores cercanos al verde (incluyendo amarillo y azul)
mascara_colores_cercanos = (
    (canal_verde > 50) |  # Verde
    ((canal_rojo > 50) & (canal_verde > 50)) |  # Amarillo
    ((canal_azul > 50) & (canal_verde > 50))  # Cian/Azul-Verde
)

# Crear una imagen para almacenar las longitudes de onda
imagen_longitudes_onda = np.zeros(imagen_rgb.shape[:2])

# Calcular la longitud de onda para cada píxel
for i in range(imagen_rgb.shape[0]):
    for j in range(imagen_rgb.shape[1]):
        if mascara_colores_cercanos[i, j]:
            r, g, b = imagen_rgb[i, j]
            wavelength = rgb_to_wavelength(r, g, b)
            imagen_longitudes_onda[i, j] = wavelength
        else:
            imagen_longitudes_onda[i, j] = np.nan  # Píxeles fuera de la máscara se visualizarán en negro

# Crear la figura y los subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Mostrar la imagen original
axes[0].imshow(imagen_rgb, aspect='auto')
axes[0].set_title('Imagen Original')

# Mostrar la imagen de longitudes de onda con el colormap 'RdYlGn' invertido
im = axes[1].imshow(imagen_longitudes_onda, cmap='summer', vmin=450, vmax=590, aspect='auto')
axes[1].set_title('Longitudes de Onda Aproximadas')

# Agregar barra de color a la segunda imagen
cbar = plt.colorbar(im, ax=axes[1], label='Longitud de Onda (nm)')

# Ajustar el layout para que no haya superposiciones
plt.tight_layout()

# Mostrar la figura
plt.show()
#%%

#podrían querer filtrar la imagen y por ejemplo solo ver lo que identifiquen como verde

# Leer la imagen
imagen = cv2.imread(r"C:\Users\Usuario\Downloads\hojas4.jpg")

# Convertir la imagen de BGR a RGB (OpenCV carga la imagen en BGR por defecto)
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Extraer los canales rojo, verde y azul
canal_rojo = imagen_rgb[:, :, 0]
canal_verde = imagen_rgb[:, :, 1]
canal_azul = imagen_rgb[:, :, 2]

# Crear una máscara donde el verde es dominante
mascara_verde = (canal_verde > canal_rojo) & (canal_verde > canal_azul)

# Aplicar la máscara a los canales para mantener solo el verde y poner los otros en negro
imagen_filtrada = np.zeros_like(imagen_rgb)
imagen_filtrada[:, :, 1] = np.where(mascara_verde, canal_verde, 0)

# Mostrar la imagen original y la imagen filtrada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(imagen_rgb)
plt.title('Imagen Original')
plt.subplot(1, 2, 2)
plt.imshow(imagen_filtrada)
plt.title('Imagen Filtrada (Solo Verde)')
plt.show()

# Diferenciar las longitudes de onda en el canal verde
# Suponiendo que el rango de intensidad va de 0 a 255 y queremos mapearlo a longitudes de onda de 500 a 570 nm
longitud_onda_min = 500
longitud_onda_max = 570
longitudes_onda = np.linspace(longitud_onda_min, longitud_onda_max, 256)

# Crear una imagen con las longitudes de onda mapeadas
imagen_longitudes_onda = np.zeros_like(imagen_filtrada, dtype=float)
imagen_longitudes_onda[:, :, 1] = longitudes_onda[imagen_filtrada[:, :, 1]]
