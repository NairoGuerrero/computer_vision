"""
MÓDULO DE MANIPULACIÓN DE VECTORES, PUNTOS, LÍNEAS Y CÓNICAS EN COORDENADAS HOMOGÉNEAS

Este módulo define clases para representar y operar con vectores homogéneos,
puntos, líneas y cónicas. Incluye métodos para cálculos geométricos como producto
punto, producto cruz, escalamiento, y construcción de cónicas a partir de puntos o líneas.
También provee métodos de graficación utilizando matplotlib.

Autores:
    - Nairo Guerrero Marquez
    - Santiago Rojas Diez
    - Juan Esteba Osorio Montoya

"""


import matplotlib.pyplot as plt
import numpy as np

class Vectorhomogeneo:
    """
    Clase base para representar vectores en coordenadas homogéneas.

    Atributos:
        vector1 (tuple): Una tupla que representa las coordenadas del vector.
    """

    def __init__(self, vector1: tuple):
        """
        Inicializa una instancia de Vectorhomogeneo.

        Args:
            vector1 (tuple): Tupla que contiene las coordenadas del vector.
        """
        self.vector1 = vector1

    def __getitem__(self, index):
        """
        Permite acceder a las coordenadas del vector mediante indexación.

        Args:
            index (int): Índice de la coordenada a obtener.

        Returns:
            Valor en la posición 'index' de la tupla vector1.
        """
        return self.vector1[index]

    def productopunto(self, vector2: tuple):
        """
        Calcula el producto punto (escalar) entre este vector y otro vector.

        Args:
            vector2 (tuple): Tupla que representa el segundo vector.

        Returns:
            float: Resultado del producto punto.
        """
        productoescalar = 0
        # Itera sobre las primeras 3 coordenadas para calcular el producto
        for i in range(3):
            productoescalar += self.vector1[i] * vector2[i]
        return productoescalar

    def escalamiento(self, k):
        """
        Escala el vector multiplicando cada coordenada por un factor k.

        Args:
            k (float): Factor de escala.

        Returns:
            tuple: Nuevo vector escalado.
        """
        # Se crea una nueva tupla con cada componente escalada
        escalado = tuple(k * i for i in self.vector1)
        return escalado

    def productocruz(self, vector2: tuple):
        """
        Calcula el producto cruz entre este vector y otro vector.

        Dependiendo del tipo de los operandos (Punto o Linea), el resultado se
        puede interpretar como una línea o un punto.

        Args:
            vector2 (tuple): Tupla que representa el segundo vector.

        Returns:
            tuple, Punto o Linea: Resultado del producto cruz.
        """
        # Cálculo de las componentes del producto cruz
        x = self.vector1[1] * vector2[2] - self.vector1[2] * vector2[1]
        y = self.vector1[2] * vector2[0] - self.vector1[0] * vector2[2]
        z = self.vector1[0] * vector2[1] - self.vector1[1] * vector2[0]
        resultado = (x, y, z)

        # Verifica los tipos para retornar el objeto adecuado
        if isinstance(self, Punto) and isinstance(vector2, Punto):
            # Producto cruz entre dos puntos devuelve una línea
            return Linea(resultado)
        elif isinstance(self, Linea) and isinstance(vector2, Linea):
            # Producto cruz entre dos líneas devuelve un punto
            return Punto(resultado)
        else:
            # Caso general, retorna la tupla de resultado
            return resultado

class Punto(Vectorhomogeneo):
    """
    Clase que representa un punto en coordenadas homogéneas, hereda de Vectorhomogeneo.
    """

    def graficar(self, ax=None, **kwargs):
        """
        Grafica el punto en un plano 2D convirtiendo de coordenadas homogéneas a cartesianas.

        Args:
            ax (matplotlib.axes.Axes, opcional): Objeto de ejes de matplotlib donde se graficará.
                Si es None, se creará una nueva figura y ejes.
            **kwargs: Argumentos adicionales para personalizar la graficación (por ejemplo, color, marcador).

        Returns:
            matplotlib.axes.Axes: Objeto de ejes con el punto graficado.

        Raises:
            ValueError: Si la coordenada 'w' es cero, lo que indica un punto en el infinito.
        """
        # Extrae las coordenadas homogéneas
        x, y, w = self.vector1
        if w == 0:
            raise ValueError("La coordenada w es cero, lo que indica que es un punto en el infinito")
        # Conversión a coordenadas cartesianas
        xcart = x / w
        ycart = y / w

        # Si no se provee un eje, se crea una nueva figura y eje
        if ax is None:
            fig, ax = plt.subplots()

        # Grafica el punto
        ax.plot(xcart, ycart, 'o', **kwargs)
        # Anota el punto con sus coordenadas
        ax.annotate(f'({xcart:.2f}, {ycart:.2f})',
                    (xcart, ycart),
                    textcoords="offset points",
                    xytext=(-4, 10),
                    ha='center',
                    fontsize=5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return ax

class Linea(Vectorhomogeneo):
    """
    Clase que representa una línea en coordenadas homogéneas, hereda de Vectorhomogeneo.
    """

    def graficar(self, ax=None, punto1=None, punto2=None, **kwargs):
        """
        Grafica la línea en un plano 2D. Calcula la pendiente e intersección a partir de la
        representación en coordenadas homogéneas y dibuja la línea.

        Args:
            ax (matplotlib.axes.Axes, opcional): Eje donde graficar. Si es None, se crea un nuevo eje.
            punto1 (tuple, opcional): Coordenadas de un punto para determinar el rango en x.
            punto2 (tuple, opcional): Coordenadas de otro punto para determinar el rango en x.
            **kwargs: Argumentos adicionales para la personalización de la gráfica.

        Returns:
            matplotlib.axes.Axes: Objeto de ejes con la línea graficada.
        """
        # Desempaqueta los coeficientes de la línea
        m, w, b = self.vector1

        # Calcula la pendiente e intersección a partir de la ecuación general de la línea
        pendiente = m / -w
        intersecto = b / -w

        # Determina el rango de valores en x a graficar, basado en los puntos proporcionados
        if punto1 is not None and punto2 is not None:
            x = (punto1[0] / punto1[2], punto2[0] / punto2[2])
            xmax = max(x)
            xmin = min(x)
        else:
            xmax = 10
            xmin = -10

        # Genera valores de x e y para graficar la línea
        xvalores = np.linspace(xmin, xmax, num=50)
        yvalores = [pendiente * x + intersecto for x in xvalores]

        # Crea el eje si no fue proporcionado
        if ax is None:
            fig, ax = plt.subplots()

        # Grafica la línea
        ax.plot(xvalores, yvalores, **kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return ax

class Conica:
    """
    Clase que representa una cónica definida por la ecuación general:
    A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0

    Atributos:
        A, B, C, D, E, F (float): Coeficientes de la ecuación de la cónica.
        matrizc (list): Matriz asociada a la cónica para cálculos adicionales.
    """

    def __init__(self, coeficientes):
        """
        Inicializa la cónica con los coeficientes dados.

        Args:
            coeficientes (iterable): Lista o tupla con los coeficientes [A, B, C, D, E, F].
        """
        self.A, self.B, self.C, self.D, self.E, self.F = coeficientes
        # Construcción de la matriz asociada a la cónica para otros cálculos (por ejemplo, la línea tangente)
        self.matrizc = [[self.A, self.B / 2, self.D / 2],
                        [self.B / 2, self.C, self.E / 2],
                        [self.D / 2, self.E / 2, self.F]]

    def construirconicapuntos(p1: Punto, p2: Punto, p3: Punto, p4: Punto, p5: Punto):
        """
        Construye una cónica que pasa por 5 puntos dados.

        Args:
            p1, p2, p3, p4, p5 (Punto): Instancias de la clase Punto.

        Returns:
            Conica: Instancia de Conica construida a partir de los 5 puntos.
        """
        # Extrae las coordenadas homogéneas de cada punto
        puntos = [p1.vector1, p2.vector1, p3.vector1, p4.vector1, p5.vector1]
        matriz = []

        # Construye la matriz de coeficientes de la ecuación de la cónica para cada punto
        for x, y, w in puntos:
            fila = [x**2, x*y, y**2, x*w, y*w, w**2]  # Coeficientes correspondientes
            matriz.append(fila)
        matriz = np.array(matriz)

        # Se utiliza la descomposición en valores singulares para resolver el sistema homogéneo
        _, _, VT = np.linalg.svd(matriz)
        coeficientes = VT[-1]  # Se toma el vector correspondiente al valor singular más pequeño

        return Conica(coeficientes)

    def construirconicalineas(l1: Linea, l2: Linea):
        """
        Construye una cónica a partir de dos líneas dadas.

        La cónica construida tiene como elementos a las líneas l1 y l2.

        Args:
            l1, l2 (Linea): Instancias de la clase Linea.

        Returns:
            Conica: Instancia de Conica construida a partir de las dos líneas.

        Raises:
            ValueError: Si las líneas son paralelas (w == 0) y no se puede obtener la cónica.
        """
        # Calcula el punto de intersección (producto cruz) de las dos líneas
        x, y, w = l1.productocruz(l2)
        if w == 0:
            raise ValueError("Líneas paralelas, no se puede obtener la cónica")

        # Construye los coeficientes de la cónica a partir de los coeficientes de las líneas
        coeficientes = [l1[0] * l2[0],
                        l1[0] * l2[1] + l2[0] * l1[1],
                        l1[1] * l2[1],
                        l1[0] * l2[2] + l2[0] * l1[2],
                        l1[1] * l2[2] + l2[1] * l1[2],
                        l1[2] * l2[2]]
        return Conica(coeficientes)

    def lineatangente(self, punto: Punto):
        """
        Calcula la línea tangente a la cónica en un punto dado.

        Args:
            punto (Punto): Punto en el cual se desea la tangente.

        Returns:
            Linea: Instancia de Linea que representa la tangente a la cónica en el punto.
        """
        resultado = []
        # Calcula el producto de la matriz de la cónica con las coordenadas del punto
        for fila in self.matrizc:
            s = 0
            for a, x in zip(fila, punto):
                s += a * x
            resultado.append(s)
        return Linea(tuple(resultado))

    def graficar(self, ax=None, rango=(-10, 10), puntos=400, **kwargs):
        """
        Grafica la cónica en un plano 2D mediante un contorno donde la ecuación se anula.

        Args:
            ax (matplotlib.axes.Axes, opcional): Eje donde se realizará la graficación.
                Si es None, se creará una nueva figura y eje.
            rango (tuple, opcional): Rango (mínimo, máximo) para los ejes x e y.
            puntos (int, opcional): Número de puntos a utilizar para la malla de graficación.
            **kwargs: Argumentos adicionales para personalizar la gráfica.

        Returns:
            matplotlib.axes.Axes: Objeto de ejes con la cónica graficada.
        """
        # Genera una malla de valores en el rango especificado
        x_vals = np.linspace(rango[0], rango[1], puntos)
        y_vals = np.linspace(rango[0], rango[1], puntos)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Evalúa la ecuación de la cónica en cada punto de la malla
        Z = (self.A * X**2 + self.B * X * Y + self.C * Y**2 +
             self.D * X + self.E * Y + self.F)

        # Crea la figura y eje si no fueron proporcionados
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        # Grafica la línea de contorno donde Z=0, que corresponde a la cónica
        ax.contour(X, Y, Z, levels=[0], colors='r', **kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return ax

# ============================================================
# EJEMPLO DE FUNCIONAMIENTO DEL MÓDULO
# ============================================================

# Se crean instancias de puntos en coordenadas homogéneas
punto1 = Punto((4, 0, 1))
punto2 = Punto((0, 2, 1))
punto3 = Punto((-4, 0, 1))
punto4 = Punto((0, -2, 1))
punto6 = Punto((2, 1.73, 1))

# Se generan líneas a partir del producto cruz de puntos
linea1 = punto1.productocruz(punto2)  # Línea que pasa por punto1 y punto2
linea2 = punto3.productocruz(punto6)  # Línea que pasa por punto3 y punto6

# Se obtiene el punto de intersección de las dos líneas
punto5 = linea1.productocruz(linea2)

# Construcción de cónicas:
# conica1 se construye a partir de 5 puntos
conica1 = Conica.construirconicapuntos(punto1, punto2, punto3, punto4, punto6)
# conica2 se construye a partir de 2 líneas
conica2 = Conica.construirconicalineas(linea1, linea2)

# Se calcula la línea tangente a la cónica en el punto6
lineatangenteconica = conica1.lineatangente(punto6)

# Se procede a graficar todos los elementos en el mismo plano
ax = punto1.graficar()         # Grafica el punto1
punto2.graficar(ax)             # Grafica el punto2 en el mismo eje
punto3.graficar(ax)             # Grafica el punto3
punto4.graficar(ax)             # Grafica el punto4
linea1.graficar(ax, punto1.vector1, punto2.vector1)  # Grafica la línea entre punto1 y punto2
linea2.graficar(ax, punto3.vector1, punto6.vector1)  # Grafica la línea entre punto3 y punto6
punto5.graficar(ax)             # Grafica el punto de intersección de las líneas
punto6.graficar(ax)             # Grafica el punto6
conica1.graficar(ax)            # Grafica la cónica construida a partir de puntos
conica2.graficar(ax)            # Grafica la cónica construida a partir de líneas
lineatangenteconica.graficar(ax) # Grafica la línea tangente a la cónica en punto6

# Muestra la figura resultante
plt.show()
