import matplotlib
import cupy as cp
matplotlib.use('TkAgg')  # Forzar uso de backend Tk
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parámetros de la simulación y del dominio
nx, ny = 512, 256       # número de celdas en X e Y
Lx, Ly = 2.0, 1.0       # dimensiones del dominio (m)
dx, dy = Lx/nx, Ly/ny   # tamaño de celda

# Parámetros físicos
rho = 1.0               # densidad
nu = 2e-3            # viscosidad cinemática 1e-2 m²/s
dt = 5e-4   # paso de tiempo      1e-3 s
nt = 1000            # número de pasos totales

# Crear mallas espaciales en la GPU
x = cp.linspace(0, Lx, nx)
y = cp.linspace(0, Ly, ny)
X, Y = cp.meshgrid(x, y)

# Inicialización de los campos: velocidades (u, v) y presión (p)
u = cp.zeros((ny, nx))
v = cp.zeros((ny, nx))
p = cp.zeros((ny, nx))

# Condición de entrada: flujo constante por la izquierda
u[:, 0] = 1.0

# Definir obstáculo: cilindro en (0.5, 0.5) con radio 0.1 m
obstacle = (X - 0.5)**2 + (Y - 0.5)**2 < 0.1**2




def build_up_b(u, v, dx, dy, dt):
    """
    Construye el término fuente 'b' para la ecuación de Poisson de la presión.
    """
    b = cp.zeros_like(u)
    b[1:-1, 1:-1] = ( rho * (1/dt * (
                        (u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx) +
                        (v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy)
                    ) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx))**2 -
                    2 * ((u[2:, 1:-1] - u[0:-2, 1:-1])/(2*dy) *
                         (v[1:-1, 2:] - v[1:-1, 0:-2])/(2*dx)) -
                    ((v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy))**2 ))
    return b

def pressure_poisson(p, dx, dy, b, nit):
    """
    Resuelve la ecuación de Poisson para la presión usando un método iterativo.
    """
    for _ in range(nit):
        p[1:-1, 1:-1] = (((p[1:-1, 2:] + p[1:-1, 0:-2])*dy**2 +
                           (p[2:, 1:-1] + p[0:-2, 1:-1])*dx**2) /
                          (2*(dx**2+dy**2))
                          - dx**2*dy**2/(2*(dx**2+dy**2)) * b[1:-1, 1:-1])
        # Condiciones de borde para la presión
        p[:, -1] = p[:, -2]   # dp/dx = 0 en el borde derecho
        p[:, 0] = p[:, 1]     # dp/dx = 0 en el borde izquierdo
        p[0, :] = p[1, :]     # dp/dy = 0 en el borde inferior
        p[-1, :] = 0          # p = 0 en el borde superior
    return p

def cavity_flow(u, v, p, dx, dy, dt, nu, obstacle):
    """
    Realiza un paso en el tiempo de la simulación de Navier-Stokes.
    """
    un = u.copy()
    vn = v.copy()

    # Paso 1: calcular la fuente para la presión
    b = build_up_b(u, v, dx, dy, dt)

    # Paso 2: resolver la ecuación de Poisson para la presión
    p = pressure_poisson(p, dx, dy, b, nit=50)

    # Paso 3: actualizar las velocidades (u y v)
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt/dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt/dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt/(2*rho*dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * (dt/dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                           dt/dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[0:-2, 1:-1])))

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt/dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt/dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                     dt/(2*rho*dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu * (dt/dx**2 * (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                           dt/dy**2 * (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

    # Condiciones de frontera
    u[:, 0] = 1.0  # Condición de entrada (flujo constante a la izquierda)
    v[:, 0] = 0.0  # Sin velocidad en la frontera izquierda

    # Paredes superior e inferior (no deslizamiento)
    u[0, :] = 0.0  # Sin velocidad en la parte superior
    u[-1, :] = 0.0  # Sin velocidad en la parte inferior
    v[0, :] = 0.0  # Sin velocidad en la parte superior
    v[-1, :] = 0.0  # Sin velocidad en la parte inferior

    # Condición de salida libre en la pared derecha
    p[:, -1] = p[:, -2]   # dp/dx = 0 en el borde derecho
    u[:, -1] = u[:, -2]   # Sin gradiente de velocidad en x en la pared derecha
    v[:, -1] = v[:, -2]   # Sin gradiente de velocidad en y en la pared derecha

    # Obstáculo (cilindro)
    u[obstacle] = 0.0
    v[obstacle] = 0.0

    return u, v, p

def compute_vorticity(u, v, dx, dy):
    """
    Calcula la vorticidad (ω = ∂v/∂x - ∂u/∂y).
    """
    vort = cp.zeros_like(u)
    vort[1:-1, 1:-1] = ((v[1:-1, 2:] - v[1:-1, 0:-2])/(2*dx) -
                        (u[2:, 1:-1] - u[0:-2, 1:-1])/(2*dy))
    return vort

# Configuración de la figura para la animación
fig, ax = plt.subplots(figsize=(8, 4))
# Se inicializa la imagen a partir de un array transferido a CPU
im = ax.imshow(cp.asnumpy(cp.zeros((ny, nx))), cmap='jet', extent=[0, Lx, 0, Ly], origin='lower')
ax.set_title('Vorticidad')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
fig.colorbar(im)

def update(frame):
    global u, v, p
    # Se realizan 10 pasos de tiempo por frame para acelerar la evolución
    for _ in range(5):
        u, v, p = cavity_flow(u, v, p, dx, dy, dt, nu, obstacle)
    vort = compute_vorticity(u, v, dx, dy)
    # Se transfiere la vorticidad a CPU para la visualización
    im.set_data(cp.asnumpy(vort))
    return [im]

ani = animation.FuncAnimation(fig, update, frames=nt, interval=1, blit=True) # 
plt.show()