import numpy as np
import matplotlib.pyplot as plt


#Solución Analítica

def solucion_analitica(t, y_inicial, v_inicial, omega, g=9.8):
    """Solución exacta de y'' = -omega**2*y - g."""
    return (
        (y_inicial + g / omega**2) * np.cos(omega * t)
        + (v_inicial / omega) * np.sin(omega * t)
        - g / omega**2
    )

#Metodo de Euler


def euler_method(f, x0, y0, h, N):
    x_values = np.linspace(x0, x0 + N * h, N + 1)
    y_values = np.empty((N + 1, len(y0)))
    y_values[0] = y0

    for i in range(N):
        y_values[i + 1] = y_values[i] + h * f(x_values[i], y_values[i])

    return x_values, y_values


# Ecuación de segundo orden: y'' = -omega^2 * y - g
# Sistema de primer orden: [y', v'] = [v, -omega^2 * y - g]
def f(t, state, omega=1.0, g=9.8):
    y, v = state
    return np.array([v, -omega**2 * y - g])


x0 = 0.0
y0 = np.array([0.0, 1.0])  # y(0)=0, v(0)=1
h = 0.1
N = 100
omega = 1.0
t_euler, y_euler = euler_method(
    lambda t, state: f(t, state, omega=omega, g=9.8),
    x0,
    y0,
    h,
    N
)


y_exacta = solucion_analitica(t_euler, y0[0], y0[1], omega)
plt.plot(t_euler, y_euler[:, 0], label=f'Euler, ω={omega}')
plt.plot(t_euler, y_exacta, '--', label=f'Analítica, ω={omega}')

plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Solución de la ecuación: Euler y analítica')
plt.legend()
plt.grid(True)
plt.show()

#Método de euler-cromer
def euler_cromer(x0, y0, v0, h, N, omega, g=9.8):

    t = np.linspace(x0, x0 + N*h, N+1)

    y = np.zeros(N+1)
    v = np.zeros(N+1)

    y[0] = y0
    v[0] = v0

    for i in range(N):
        a = -omega**2 * y[i] - g

        v[i+1] = v[i] + h*a

        y[i+1] = y[i] + h*v[i+1]

    return t, y, v


# Condiciones iniciales
x0 = 0.0
y0 = 0.0
v0 = 1.0
h = 0.1
N = 100
omega = 1.0


t, y_num, v_num = euler_cromer(
    x0,
    y0,
    v0,
    h,
    N,
    omega
)

# Solución exacta
y_exacta = solucion_analitica(
    t,
    y0,
    v0,
    omega
)

# Gráficas
plt.plot(t, y_num, label='Euler-Cromer')
plt.plot(t, y_exacta, '--', label='Analítica')

plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Euler-Cromer vs Solución Analítica')
plt.legend()
plt.grid(True)
plt.show()

#Método salto de rana:

def leapfrog(y0, v0, h, N, omega, g=9.8):

    y = np.zeros(N+1)
    v = np.zeros(N+1)

    y[0] = y0
    v[0] = v0

    for i in range(N):
        a = -omega**2 * y[i] - g
        v_half = v[i] + 0.5 * h * a
        y[i+1] = y[i] + h * v_half
        a_siguiente = -omega**2 * y[i+1] - g
        v[i+1] = v_half + 0.5 * h * a_siguiente

    return y, v

# Solución por salto de rana y solución analítica
y_rana, v_rana = leapfrog(y0, v0, h, N, omega)
t_rana = np.linspace(x0, x0 + N * h, N + 1)
y_exacta_rana = solucion_analitica(t_rana, y0, v0, omega)

# Gráfica comparación
plt.figure(figsize=(8, 5))

plt.plot(t_rana, y_rana, label='Salto de rana')
plt.plot(t_rana, y_exacta_rana, '--', label='Analítica')

plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Leapfrog vs Solución Analítica')

plt.legend()
plt.grid(True)

plt.show()

#Método de Rugen-Kutta 2do orden:

def heun_method(f, x0, y0, h, N):

    x_values = np.linspace(x0, x0 + N*h, N+1)

    y_values = np.empty((N+1, len(y0)))

    y_values[0] = y0

    for i in range(N):

        # Pendiente en el punto actual
        k1 = f(x_values[i], y_values[i])

        # Predicción tipo Euler
        y_pred = y_values[i] + h*k1

        # Pendiente en el punto predicho
        k2 = f(x_values[i] + h, y_pred)

        # Corrección de Heun
        y_values[i+1] = y_values[i] + (h/2)*(k1 + k2)

    return x_values, y_values

# Aplicar Heun al sistema de posición y velocidad
y0_rk = np.array([y0, v0])
t_heun, y_heun = heun_method(
    lambda t, state: f(t, state, omega=omega),
    x0,
    y0_rk,
    h,
    N
)
y_exacta_rk = solucion_analitica(t_heun, y0, v0, omega)

plt.figure(figsize=(8, 5))
plt.plot(t_heun, y_heun[:, 0], label='Runge-Kutta 2 (Heun)')
plt.plot(t_heun, y_exacta_rk, '--', label='Analítica')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Método de Heun vs Solución Analítica')
plt.legend()
plt.grid(True)
plt.show()

#Método de Rugen-Kutta de 4to orden

def rk4_method(f, x0, y0, h, N):

    x_values = np.linspace(x0, x0 + N*h, N+1)

    y_values = np.empty((N+1, len(y0)))

    y_values[0] = y0

    for i in range(N):

        k1 = f(x_values[i], y_values[i])

        k2 = f(
            x_values[i] + h/2,
            y_values[i] + (h/2)*k1
        )

        k3 = f(
            x_values[i] + h/2,
            y_values[i] + (h/2)*k2
        )

        k4 = f(
            x_values[i] + h,
            y_values[i] + h*k3
        )

        y_values[i+1] = (
            y_values[i]
            + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        )

    return x_values, y_values

# Ejecutar RK4 para el sistema de posición y velocidad
y0_rk4 = np.array([y0, v0])
t_rk4, y_rk4 = rk4_method(
    lambda t, state: f(t, state, omega=omega),
    x0,
    y0_rk4,
    h,
    N
)
y_exacta_rk4 = solucion_analitica(t_rk4, y0, v0, omega)

# Gráfica comparación
plt.figure(figsize=(8,5))

plt.plot(
    t_rk4,
    y_rk4[:,0],
    label='Runge-Kutta 4'
)

plt.plot(
    t_rk4,
    y_exacta_rk4,
    '--',
    label='Analítica'
)

plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('RK4 vs Solución Analítica')

plt.legend()
plt.grid(True)

plt.show()

#Análisis de energia de cada método:
# Energia cinetica + energia elastica + energia gravitatoria
def energia(y, v, omega, g=9.8, m=1.0):
    return 0.5 * m * v**2 + 0.5 * m * omega**2 * y**2 + m * g * y


E_euler = energia(y_euler[:, 0], y_euler[:, 1], omega)
E_ec = energia(y_num, v_num, omega)
E_verlet = energia(y_rana, v_rana, omega)
E_rk2 = energia(y_heun[:, 0], y_heun[:, 1], omega)
E_rk4 = energia(y_rk4[:, 0], y_rk4[:, 1], omega)

plt.figure(figsize=(11, 6))
plt.ticklabel_format(useOffset=False)

plt.plot(t_euler, E_euler, label="Euler")
plt.plot(t, E_ec, label="Euler-Cromer")
plt.plot(t_rana, E_verlet, label="Salto de rana (Verlet)")
plt.plot(t_heun, E_rk2, label="RK2")
plt.plot(t_rk4, E_rk4, "--", label="RK4")

plt.xlabel("Tiempo (s)")
plt.ylabel("Energia mecanica (J)")
plt.title("Evolucion de la energia del sistema")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
