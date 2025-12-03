import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from scipy import special

class DiffusionSimulator2D:
    """
    2D симулятор диффузии примесей и точечных дефектов в кремнии
    """
    
    def __init__(self, nx, ny, dx, dy):
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.C_i = np.zeros((ny, nx))  # концентрация примеси
        self.C_I = np.zeros((ny, nx))  # концентрация междоузлий
        self.C_V = np.zeros((ny, nx))  # концентрация вакансий
        
    def set_initial_conditions(self, C_i0, C_I0, C_V0):
        """Установка начальных условий"""
        self.C_i = C_i0.copy()
        self.C_I = C_I0.copy()
        self.C_V = C_V0.copy()
    
    def compute_diffusion_coefficient(self, D_i_star, f_I, C_I_eq, C_V_eq):
        """Вычисление коэффициента диффузии с учетом точечных дефектов"""
        return D_i_star * (f_I * self.C_I/C_I_eq + (1 - f_I) * self.C_V/C_V_eq)
    
    def solve_defect_equations(self, dt, D_I, D_V, K_R, C_I_eq, C_V_eq, g_I=0):
        """
        Решение уравнений для точечных дефектов методом конечных разностей
        """
        # Преобразуем g_I в массив правильной размерности если это скаляр
        if np.isscalar(g_I):
            g_I_array = np.full((self.ny, self.nx), g_I)
        else:
            g_I_array = g_I
            
        # Вычисляем члены рекомбинации
        recombination = K_R * (self.C_I * self.C_V - C_I_eq * C_V_eq)
        
        # Неявная схема для междоузлий
        A_I = self.build_matrix(D_I, dt, self.dx, self.dy)
        b_I = self.C_I.ravel() + dt * (g_I_array.ravel() - recombination.ravel())
        self.C_I = spsolve(A_I, b_I).reshape(self.ny, self.nx)
        
        # Неявная схема для вакансий
        A_V = self.build_matrix(D_V, dt, self.dx, self.dy)
        b_V = self.C_V.ravel() + dt * (-recombination.ravel())
        self.C_V = spsolve(A_V, b_V).reshape(self.ny, self.nx)
    
    def solve_impurity_diffusion(self, dt, D_i, z, T, epsilon):
        """
        Решение уравнения диффузии примеси с учетом электрического поля
        """
        # Если D_i - скаляр, преобразуем в массив
        if np.isscalar(D_i):
            D_i_array = np.full((self.ny, self.nx), D_i)
        else:
            D_i_array = D_i
            
        # Вычисление электрического поля через градиент концентрации
        grad_C_x, grad_C_y = np.gradient(self.C_i, self.dx, self.dy)
        
        # Избегаем деления на ноль
        denominator = epsilon + np.abs(self.C_i)
        E_x = -grad_C_x / denominator
        E_y = -grad_C_y / denominator
        
        # Построение матрицы для уравнения диффузии
        # Используем среднее значение D_i для построения матрицы
        D_avg = np.mean(D_i_array)
        A_i = self.build_matrix(D_avg, dt, self.dx, self.dy)
        
        # Добавление дрейфового члена
        kT = 1.38e-23 * T / 1.6e-19  # kT в эВ
        drift_x = (z * D_i_array / kT) * (E_x * grad_C_x)
        drift_y = (z * D_i_array / kT) * (E_y * grad_C_y)
        drift_term = drift_x + drift_y
        
        b_i = self.C_i.ravel() + dt * drift_term.ravel()
        self.C_i = spsolve(A_i, b_i).reshape(self.ny, self.nx)
    
    def build_matrix(self, D, dt, dx, dy):
        """
        Построение матрицы для неявной схемы
        """
        n = self.nx * self.ny
        A = sparse.lil_matrix((n, n))
        
        alpha_x = D * dt / dx**2
        alpha_y = D * dt / dy**2
        
        for i in range(self.ny):
            for j in range(self.nx):
                idx = i * self.nx + j
                A[idx, idx] = 1 + 2 * alpha_x + 2 * alpha_y
                
                # Соседи по x
                if j > 0:
                    A[idx, idx-1] = -alpha_x
                if j < self.nx-1:
                    A[idx, idx+1] = -alpha_x
                
                # Соседи по y
                if i > 0:
                    A[idx, idx-self.nx] = -alpha_y
                if i < self.ny-1:
                    A[idx, idx+self.nx] = -alpha_y
        
        return A.tocsr()
    
    def oxidation_model(self, y_positions, t_ox, params):
        """
        Модель окисления - формирование 'птичьего клюва'
        """
        a1, b1, c1, d1 = params['a1'], params['b1'], params['c1'], params['d1']
        a2, b2, c2, d2 = params['a2'], params['b2'], params['c2'], params['d2']
        
        Z1 = a1 * special.erfc(b1 * y_positions + c1) + d1
        Z2 = a2 * special.erfc(b2 * y_positions + c2) + d2
        
        return Z1, Z2

    def apply_boundary_conditions(self, bc_type='zero_flux'):
        """
        Применение граничных условий
        """
        if bc_type == 'zero_flux':
            # Нулевой поток на границах (симметричные условия)
            self.C_i[0, :] = self.C_i[1, :]   # верхняя граница
            self.C_i[-1, :] = self.C_i[-2, :] # нижняя граница  
            self.C_i[:, 0] = self.C_i[:, 1]   # левая граница
            self.C_i[:, -1] = self.C_i[:, -2] # правая граница
            
            self.C_I[0, :] = self.C_I[1, :]
            self.C_I[-1, :] = self.C_I[-2, :]
            self.C_I[:, 0] = self.C_I[:, 1]
            self.C_I[:, -1] = self.C_I[:, -2]
            
            self.C_V[0, :] = self.C_V[1, :]
            self.C_V[-1, :] = self.C_V[-2, :]
            self.C_V[:, 0] = self.C_V[:, 1]
            self.C_V[:, -1] = self.C_V[:, -2]



# Пример использования симулятора
def example_simulation():
    # Параметры сетки
    nx, ny = 50, 50  # Уменьшим размер для скорости
    dx, dy = 2e-7, 2e-7  # 0.2 μm
    
    # Создание симулятора
    simulator = DiffusionSimulator2D(nx, ny, dx, dy)
    
    # Начальные условия - Gaussian профиль
    x = np.linspace(0, nx*dx, nx)
    y = np.linspace(0, ny*dy, ny)
    X, Y = np.meshgrid(x, y)
    
    # Центр распределения
    x0, y0 = nx*dx/2, ny*dy/2
    
    C_i0 = 1e18 * np.exp(-((X-x0)**2 + (Y-y0)**2) / (2e-6)**2)
    C_I0 = np.ones((ny, nx)) * 1e10  # равновесная концентрация
    C_V0 = np.ones((ny, nx)) * 1e10  # равновесная концентрация
    
    simulator.set_initial_conditions(C_i0, C_I0, C_V0)
    
    # Параметры материалов (более реалистичные значения)
    params = {
        'D_i_star': 1e-14,    # см²/с
        'f_I': 0.5,           # доля междоузельного механизма
        'D_I': 1e-6,          # диффузия междоузлий
        'D_V': 1e-7,          # диффузия вакансий  
        'K_R': 1e-21,         # константа рекомбинации
        'C_I_eq': 1e10,       # равновесная концентрация междоузлий
        'C_V_eq': 1e10,       # равновесная концентрация вакансий
        'T': 1100 + 273,      # температура в К
        'z': -1               # донорная примесь
    }
    
    # Временной цикл
    dt = 0.01
    n_steps = 50
    
    print("Запуск симуляции...")
    for step in range(n_steps):
        if step % 10 == 0:
            print(f"Шаг {step}/{n_steps}")
            
        # Вычисление коэффициента диффузии
        D_i = simulator.compute_diffusion_coefficient(
            params['D_i_star'], params['f_I'], 
            params['C_I_eq'], params['C_V_eq']
        )
        
        # Решение уравнений для точечных дефектов
        simulator.solve_defect_equations(
            dt, params['D_I'], params['D_V'], params['K_R'],
            params['C_I_eq'], params['C_V_eq']
        )
        
        # Решение уравнения для примеси
        simulator.solve_impurity_diffusion(
            dt, D_i, params['z'], params['T'], 1e-10
        )
        
        # Применение граничных условий
        simulator.apply_boundary_conditions()
    
    # Визуализация результатов
    plt.figure(figsize=(5, 15))
    
    plt.subplot(311)
    plt.contourf(X*1e6, Y*1e6, simulator.C_i, levels=50)
    plt.title('Концентрация примеси')
    plt.xlabel('x (мкм)')
    plt.ylabel('y (мкм)')
    plt.colorbar()
    
    plt.subplot(312)  
    plt.contourf(X*1e6, Y*1e6, simulator.C_I, levels=50)
    plt.title('Концентрация междоузлий')
    plt.xlabel('x (мкм)')
    plt.ylabel('y (мкм)')
    plt.colorbar()
    
    plt.subplot(313)
    plt.contourf(X*1e6, Y*1e6, simulator.C_V, levels=50)
    plt.title('Концентрация вакансий')
    plt.xlabel('x (мкм)')
    plt.ylabel('y (мкм)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    print("Симуляция завершена успешно!")

# Запуск примера
if __name__ == "__main__":
    example_simulation()    