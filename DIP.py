"""
DIP.py

Este módulo analiza una imagen binaria para detectar y caracterizar objetos utilizando procesamiento de imágenes.
Se aplican técnicas como umbralización, operaciones morfológicas, análisis de componentes conexas y cálculo de
características geométricas. El resultado incluye una visualización con bounding boxes, elipses ajustadas y centroides.

Requiere las siguientes bibliotecas:
- OpenCV (cv2)
- NumPy
- scikit-image
- matplotlib
- math

Funcionalidad:
- Detecta objetos en una imagen binaria invertida.
- Calcula características geométricas como área, perímetro, centroide, orientación, excentricidad, etc.
- Ajusta rectángulos rotados y elipses para representar cada objeto.
- Visualiza los resultados con anotaciones gráficas sobre la imagen original.

Autores:
    - Nairo Guerrero Marquez
    - Santiago Rojas Diez
    - Juan David Perdomo

"""

import cv2
import numpy as np
from skimage.measure import regionprops, label
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
import math


def analizar_imagen(ruta_imagen):
    """
    Analiza una imagen binaria para detectar objetos y calcular sus propiedades geométricas y morfológicas.

    Parámetros:
    ruta_imagen (str): Ruta al archivo de imagen a analizar.

    Procesos realizados:
    - Conversión a escala de grises.
    - Umbralización para binarización.
    - Operación morfológica de cierre.
    - Etiquetado de componentes conexas.
    - Extracción de características para cada región válida.
    - Visualización de resultados incluyendo contornos, elipses y bounding boxes.

    Muestra:
    - Imagen original.
    - Imagen binarizada invertida.
    - Imagen con anotaciones de los objetos detectados.
    """

    # Cargar la imagen desde la ruta especificada
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print("No se pudo cargar la imagen")
        return

    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral para convertir la imagen en binaria (los objetos son oscuros sobre fondo claro)
    _, binaria = cv2.threshold(gris, 253, 255, cv2.THRESH_BINARY)

    # Invertir la imagen binaria: ahora los objetos son blancos sobre fondo negro
    binaria = cv2.bitwise_not(binaria)

    # Aplicar una operación de cierre morfológico para conectar líneas discontínuas
    kernel = np.ones((7, 7), np.uint8)
    cerrada = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Etiquetar componentes conexas en la imagen binaria cerrada
    labeled_image = label(cerrada)
    regions = regionprops(labeled_image)

    # Definir área mínima para considerar un objeto como válido
    min_area = 500
    output = imagen.copy()  # Copia de la imagen original para dibujar resultados
    objetos_validos = 0  # Contador de objetos válidos detectados

    # Iterar sobre cada región detectada
    for i, region in enumerate(regions):
        if region.area < min_area:
            continue  # Ignorar objetos pequeños

        objetos_validos += 1
        print(f"\n=== Objeto {objetos_validos} ===")

        # Extraer características geométricas básicas
        area = region.area
        perimetro = region.perimeter
        centroide = region.centroid
        bbox = region.bbox
        excentricidad = region.eccentricity
        orientacion = math.degrees(region.orientation)
        relacion_aspecto = region.major_axis_length / (region.minor_axis_length + 1e-6)
        extension = region.extent

        # Calcular características relacionadas con la forma convexa
        convex_image = convex_hull_image(region.image)
        area_convexa = np.sum(convex_image)
        solidez = region.solidity

        # Obtener rectángulo rotado que ajusta la región
        rect = cv2.minAreaRect(region.coords[:, [1, 0]])
        (x, y), (w, h), angulo_rect = rect
        box_points = cv2.boxPoints(rect)
        box_points = np.intp(box_points)

        # Intentar ajustar una elipse a los puntos del contorno si la región es suficientemente grande
        elipse_valida = False
        if region.image.shape[0] >= 5 and region.image.shape[1] >= 5:
            try:
                (x_elipse, y_elipse), (MA, ma), angle_elipse = cv2.fitEllipse(region.coords[:, [1, 0]])
                elipse_valida = True
            except:
                pass

        # Imprimir las características extraídas para el objeto
        print(f"Área: {area} píxeles")
        print(f"Perímetro: {perimetro:.2f} píxeles")
        print(f"Centroide: (x={centroide[1]:.2f}, y={centroide[0]:.2f})")
        print(f"Bounding Box: (y1={bbox[0]}, x1={bbox[1]}, y2={bbox[2]}, x2={bbox[3]})")
        print(f"Excentricidad: {excentricidad:.3f} (0=círculo, 1=linea)")
        print(f"Orientación: {orientacion:.2f}° (ángulo del eje mayor)")
        print(f"Relación de aspecto: {relacion_aspecto:.2f} (eje mayor/menor)")
        print(f"Extensión: {extension:.3f} (área/área bbox)")
        print(f"Área convexa: {area_convexa} píxeles")
        print(f"Solidez: {solidez:.3f} (área/área convexa)")
        print(f"Rectángulo rotado: centro=({x:.2f}, {y:.2f}), ancho={w:.2f}, alto={h:.2f}, ángulo={angulo_rect:.2f}°")

        if elipse_valida:
            print(f"Elipse ajustada: centro=({x_elipse:.2f}, {y_elipse:.2f}), ")
            print(f"                eje mayor={MA:.2f}, eje menor={ma:.2f}, ángulo={angle_elipse:.2f}°")
        else:
            print("Elipse ajustada: No disponible")

        # Dibujar bounding box, rectángulo rotado, elipse y centroide sobre la imagen
        minr, minc, maxr, maxc = bbox
        cv2.rectangle(output, (minc, minr), (maxc, maxr), (255, 0, 0), 2)  # Azul
        cv2.drawContours(output, [box_points], 0, (0, 255, 0), 2)  # Verde
        cv2.circle(output, (int(centroide[1]), int(centroide[0])), 5, (0, 0, 255), -1)  # Rojo

        if elipse_valida:
            cv2.ellipse(output, (int(x_elipse), int(y_elipse)),
                        (int(MA / 2), int(ma / 2)), int(angle_elipse),
                        0, 360, (255, 255, 0), 2)  # Amarillo

        # Etiquetar el objeto con su número
        cv2.putText(output, str(objetos_validos), (int(centroide[1]), int(centroide[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    print(f"\nTotal de objetos detectados: {objetos_validos}")

    # Visualización de resultados en tres paneles
    plt.figure(figsize=(15, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title('Original')

    plt.subplot(1, 3, 2)
    plt.imshow(binaria, cmap='gray')
    plt.title('Binarizada')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title(f'Objetos Detectados: {objetos_validos}')

    plt.tight_layout()
    plt.show()


# Ejecutar el análisis con una imagen de ejemplo
analizar_imagen('imagen.jpg')
