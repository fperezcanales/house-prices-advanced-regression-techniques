import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

class SVM:
    def __init__(self, C=1.0, sigma=1.0):
        self.C = C
        self.sigma = sigma
        self.alfa = None
        self.b = 0
        self.X = None
        self.y = None
        self.K = None
        self.errors = None
        self.support_vectors = []
        self.animation_frames_data = []

    def kernel_rbf(self, x_i, x_j):
        return np.exp(-np.linalg.norm(x_i - x_j) ** 2 / (2 * self.sigma ** 2))

    def calculate_kernel_matrix(self, X):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel_rbf(X[i], X[j])
        return K

    def f(self, x_i):
        result = 0.0
        if self.alfa is None or len(self.alfa) == 0:
            return self.b # Retorna b si no hay alfas (al principio)

        for i in range(len(self.alfa)):
            result += self.alfa[i] * self.y[i] * self.kernel_rbf(self.X[i], x_i)
        return result + self.b

    def error(self, i):
        return self.f(self.X[i]) - self.y[i]

    def initialize(self, X, y):
        n = X.shape[0]
        self.X = X
        self.y = y
        self.alfa = np.zeros(n)
        self.b = 0.0
        self.K = self.calculate_kernel_matrix(X)
        self.errors = np.zeros(n)
        for i in range(n):
            self.errors[i] = self.error(i)

    def select_indices(self, n):
        # Primero busca un violador KKT
        # Este bucle ahora itera sobre los errores en un orden aleatorio para evitar ciclos
        # y ayudar a la convergencia en casos difíciles.
        # Originalmente, tu código seleccionaba el primero que encontraba.
        # Aquí lo mantendremos simple como lo tienes, buscando el primero.
        # Una implementación más robusta de SMO usaría una heurística más compleja.

        # Iterar sobre todos los puntos para encontrar el primero que viola las condiciones KKT
        i_violator = -1
        for k in range(n):
            if not self.is_kkt(k):
                i_violator = k
                break

        if i_violator == -1: # No se encontraron violadores KKT
            return None

        # Luego, busca el segundo multiplicador alfa j que maximice |E1 - E2|
        max_diff = -np.inf
        j = -1
        E1 = self.errors[i_violator]

        for k in range(n):
            if k != i_violator:
                E2 = self.errors[k]
                diff = np.abs(E1 - E2)
                if diff > max_diff:
                    max_diff = diff
                    j = k

        # Si no se encontró un j válido (solo hay un punto o no se pudo mejorar)
        if j == -1:
            return None

        return i_violator, j

    def is_kkt(self, i):
        fxi = self.f(self.X[i])
        # KKT conditions with a small tolerance
        if self.alfa[i] == 0:
            return self.y[i] * fxi >= 1 - 0.001
        elif 0 < self.alfa[i] < self.C:
            return np.abs(self.y[i] * fxi - 1) < 0.001
        elif self.alfa[i] == self.C:
            return self.y[i] * fxi <= 1 + 0.001
        return False

    def partial_optimize(self, X, y, tol=0.001):
        n = X.shape[0]
        changes = 0

        # En una implementación estándar de SMO, este bucle principal sería
        # para iterar sobre los multiplicadores alfa que NO están en los límites (0 < alfa < C)
        # y luego sobre todos los multiplicadores alfa, pero tu enfoque de 'n' intentos es válido.
        # Lo más importante es que select_indices devuelva un par (i,j) válido

        indices = self.select_indices(n)
        if indices is None:
            return False # No KKT violators or no suitable pair found

        i, j = indices
        E1 = self.errors[i]
        E2 = self.errors[j]

        eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
        if eta >= 0:
            return False # No se puede reducir la función objetivo para este par.

        alpha_i_old = self.alfa[i]
        alpha_j_old = self.alfa[j]

        # Calculate L and H for alpha_j clipping (hay un error en el código original, las restricciones son para alpha_j)
        if self.y[i] == self.y[j]:
            L = max(0, self.alfa[j] + self.alfa[i] - self.C)
            H = min(self.C, self.alfa[j] + self.alfa[i])
        else:
            L = max(0, self.alfa[j] - self.alfa[i])
            H = min(self.C, self.C + self.alfa[j] - self.alfa[i])

        if L == H:
            return False

        # Actualiza alpha_j primero (tradicionalmente, es el segundo índice que se actualiza directamente)
        self.alfa[j] -= self.y[j] * (E1 - E2) / eta
        self.alfa[j] = np.clip(self.alfa[j], L, H)

        if np.abs(self.alfa[j] - alpha_j_old) < tol:
            return False # No hubo cambio significativo en alpha_j

        # Actualiza alpha_i basado en el cambio de alpha_j
        self.alfa[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alfa[j])

        # Recalculate bias
        b1 = self.b - E1 - self.y[i] * (self.alfa[i] - alpha_i_old) * self.K[i, i] - self.y[j] * (self.alfa[j] - alpha_j_old) * self.K[i, j]
        b2 = self.b - E2 - self.y[i] * (self.alfa[i] - alpha_i_old) * self.K[i, j] - self.y[j] * (self.alfa[j] - alpha_j_old) * self.K[j, j]

        if 0 < self.alfa[i] < self.C:
            self.b = b1
        elif 0 < self.alfa[j] < self.C:
            self.b = b2
        else: # Both are at bounds, choose midpoint
            self.b = (b1 + b2) / 2

        # Actualiza el caché de errores
        # Para eficiencia, solo actualizar los errores que cambian.
        # Sin embargo, para visualización y depuración, recomputar todos es seguro.
        for k_err in range(n):
            self.errors[k_err] = self.f(self.X[k_err]) - self.y[k_err]

        # Solo si hubo cambios significativos
        changes = 1
        self.support_vectors = [k for k in range(n) if self.alfa[k] > 0]
        return changes > 0

    def train(self, X, y, max_iter=1000, tol=0.001, max_passes_without_change=5, frames_per_step=1):
        n = X.shape[0]
        self.initialize(X, y)

        iter_count = 0
        passes_without_change_count = 0

        while iter_count < max_iter and passes_without_change_count < max_passes_without_change:
            # Guarda el estado ANTES de realizar los pasos de optimización para este frame
            self.animation_frames_data.append({
                'alfa': np.copy(self.alfa),
                'b': self.b,
                'support_vectors': list(self.support_vectors),
                'w_vector': self.calculate_w() # Guarda el vector w también
            })

            current_step_changes = 0
            for _ in range(frames_per_step):
                made_changes = self.partial_optimize(X, y, tol)
                if made_changes:
                    current_step_changes += 1

            if current_step_changes == 0:
                passes_without_change_count += 1
            else:
                passes_without_change_count = 0

            iter_count += 1

        # Guarda el estado final
        self.animation_frames_data.append({
            'alfa': np.copy(self.alfa),
            'b': self.b,
            'support_vectors': list(self.support_vectors),
            'w_vector': self.calculate_w()
        })

    def calculate_w(self):
        # w = sum(alfa_i * y_i * x_i) para los vectores de soporte
        if self.alfa is None or len(self.alfa) == 0:
            return np.zeros(self.X.shape[1]) # Retorna un vector cero si no hay alfas

        w = np.zeros(self.X.shape[1]) # Asumiendo 2 dimensiones para x
        for i in range(len(self.alfa)):
            if self.alfa[i] > 0: # Solo los vectores de soporte contribuyen
                w += self.alfa[i] * self.y[i] * self.X[i]
        return w


def plot_decision_boundary(X, y, model, frame_num, scatter_plot, ax):
    ax.clear()
    ax.set_xlim([np.min(X[:, 0]) - 0.5, np.max(X[:, 0]) + 0.5])
    ax.set_ylim([np.min(X[:, 1]) - 0.5, np.max(X[:, 1]) + 0.5])
    ax.set_aspect('equal', adjustable='box') # Para que el vector w sea perpendicular visualmente

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Asegúrate de que el modelo tenga alfas y b inicializados antes de llamar a f()
    if model.alfa is not None and len(model.alfa) > 0:
        Z = np.array([model.f(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.75, cmap=plt.cm.RdBu)
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'], linestyles=['--', '-', '--'])
    else:
        # Dibujar un fondo neutral si el modelo aún no está inicializado (primer frame)
        ax.contourf(xx, yy, np.zeros_like(xx), levels=[-1, 0, 1], alpha=0.75, cmap=plt.cm.RdBu)


    # Mostrar puntos de entrenamiento
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=50, edgecolors='k')

    # Resaltar vectores de soporte
    if model.support_vectors and len(model.support_vectors) > 0:
        sv_X = X[model.support_vectors]
        ax.scatter(sv_X[:, 0], sv_X[:, 1], facecolors='none', edgecolors='green', s=150, linewidth=2)

    # Dibuja el vector w (normal a la línea de decisión)
    # Lo tomamos de los datos del frame actual
    w_vec = model.animation_frames_data[frame_num]['w_vector'] # O frame_data['w_vector']
    if w_vec is not None and np.linalg.norm(w_vec) > 1e-6: # Evita dibujar si w es cero o casi cero
        # Puedes dibujar w desde el centro de los datos o desde el origen
        # Para el propósito de visualización del concepto, desde el origen está bien
        # o puedes centrarlo en el plot

        # Para centrar la flecha en la región de plot, podrías usar el centroide del plot
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Normaliza w_vec para controlar su longitud visual si lo deseas
        # o dibújalo tal cual si su magnitud es importante para la escala del margen
        # Para este ejemplo, lo dibujaremos tal cual para que refleje la magnitud 1/||w||

        # Dibuja el vector w desde un punto de referencia (ej. (center_x, center_y))
        # Puedes ajustar el punto de inicio de la flecha
        # Aquí lo dibujaremos desde el centro del área del plot
        ax.arrow(center_x, center_y, w_vec[0]*0.5, w_vec[1]*0.5, # Escalar w_vec para que no sea demasiado largo
                 head_width=0.1, head_length=0.1, fc='purple', ec='purple', linewidth=2,
                 label='Vector w (Normal)')
        ax.text(center_x + w_vec[0]*0.5 + 0.1, center_y + w_vec[1]*0.5 + 0.1,
                'w', color='purple', fontsize=12)

    ax.set_title(f"Iteración: {frame_num}")
    return ax.collections + ax.patches # Retorna todos los objetos para blitting si fuera necesario (aunque blit=False)

def animate_svm(X, y, model):
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame_num):
        if not model.animation_frames_data:
            print(f"Warning: No animation data for frame {frame_num}. Check SVM.train().")
            return []

        # Asegúrate de que el frame_num no exceda el índice de la lista
        if frame_num >= len(model.animation_frames_data):
            frame_num = len(model.animation_frames_data) - 1 # Usa el último frame disponible

        frame_data = model.animation_frames_data[frame_num]

        # Actualiza el estado del modelo para este frame
        model.alfa = frame_data['alfa']
        model.b = frame_data['b']
        model.support_vectors = frame_data['support_vectors']
        # El vector 'w_vector' se usa directamente en plot_decision_boundary

        return plot_decision_boundary(X, y, model, frame_num, None, ax)

    num_frames = len(model.animation_frames_data)
    if num_frames == 0:
        print("No frames were generated. Check your training parameters (max_iter, frames_per_step).")
        return

    # Ajusta el intervalo (milisegundos por frame) para controlar la velocidad
    ani = FuncAnimation(fig, update, frames=num_frames, repeat=False, blit=False, interval=50) # 50ms por frame = 20 fps
    ani.save('svm_training_with_w_2.gif', writer='pillow', fps=10) # 10 fps para el GIF final

# --- Datos y Ejecución ---

# Generar datos más desafiantes (las medias lunas)
def generate_realistic_data(n_samples=200, noise=0.1, random_seed=42):
    np.random.seed(random_seed)

    X = []
    y = []

    # Clase 1 (media luna superior)
    theta = np.linspace(-np.pi/2, np.pi/2, n_samples // 2)
    r = 1 + np.random.rand(n_samples // 2) * noise * 3.0 # Aumentar MUCHO el ruido en radio
    X1 = np.array([r * np.cos(theta), r * np.sin(theta) + 0.2]).T # Centrado en Y ligeramente arriba, menos separación inicial
    y1 = np.ones(n_samples // 2)

    # Clase -1 (media luna inferior)
    theta = np.linspace(np.pi/2, 3*np.pi/2, n_samples // 2)
    r = 1 + np.random.rand(n_samples // 2) * noise * 3.0 # Aumentar MUCHO el ruido en radio
    X2 = np.array([r * np.cos(theta), r * np.sin(theta) - 0.2]).T # Centrado en Y ligeramente abajo, menos separación inicial
    y2 = -np.ones(n_samples // 2)

    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])

    # Add even more overall noise to entire dataset for more challenging separation
    X += np.random.randn(*X.shape) * noise * 1.2 # Aumentar MUCHO el ruido general

    return X, y

# Generar datos
X, y = generate_realistic_data(n_samples=500, noise=0.4, random_seed=42) # Aumentar el ruido general en la llamada


# Crear y entrenar el modelo SVM con los nuevos datos
# ¡CRUCIAL: frames_per_step=1 para ver la granularidad!
# Ajusta sigma para que funcione bien con los datos de media luna
svm = SVM(C=1.0, sigma=0.5)
svm.train(X, y, max_iter=500, frames_per_step=1) # Más iteraciones y granularidad por frame

# Crear la animación
animate_svm(X, y, svm)

plt.show()