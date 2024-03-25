from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def generate_displacement_map(image_path):
    # Carregar a imagem
    input_image = Image.open(image_path).convert("L")  # Converte para escala de cinza

    # Converter imagem PIL para array numpy
    input_array = np.array(input_image, dtype=np.float32) / 255.0  # Normaliza os valores de pixel para o intervalo [0, 1]

    # Criar mapa de deslocamento
    displacement_map = np.zeros((input_image.size[1], input_image.size[0], 2), dtype=np.float32)
    displacement_map[:, :, 0] = input_array
    displacement_map[:, :, 1] = input_array

    return displacement_map

def apply_displacement_map(surface_points, displacement_map, scale=1.0):
    """
    Aplica o mapeamento de deslocamento a uma superfície.

    Args:
        surface_points: Array numpy das coordenadas dos pontos na superfície (shape: (n, 3))
        displacement_map: Mapa de deslocamento (shape: (h, w, 2))
        scale: Fator de escala para ajustar a intensidade do deslocamento

    Returns:
        surface_points_desplaced: Coordenadas dos pontos com o mapeamento de deslocamento aplicado
    """
    # Normaliza as coordenadas dos pontos da superfície para o intervalo [0, 1] para corresponder ao tamanho do mapa de deslocamento
    normalized_points = surface_points[:, :2] - np.min(surface_points[:, :2], axis=0)
    normalized_points /= np.max(surface_points[:, :2], axis=0)

    # Interpola os valores do mapa de deslocamento para obter os deslocamentos correspondentes
    interpolated_values = np.array([np.interp(normalized_points[:, 1], np.linspace(0, 1, displacement_map.shape[0]), displacement_map[:, :, i].flatten()) for i in range(2)]).T

    # Aplica os deslocamentos aos pontos na superfície
    surface_points_desplaced = surface_points.copy()
    surface_points_desplaced[:, 2] += interpolated_values[:, 0] * scale  # Deslocamento no eixo z
    surface_points_desplaced[:, :2] += interpolated_values[:, 1:] * scale  # Deslocamento nos eixos x e y

    return surface_points_desplaced

# Caminho para a imagem
#image_path = "disp_mapping_1.jpg"

# Gerar o mapa de deslocamento
#displacement_map = generate_displacement_map(image_path)

#print(displacement_map)



def ray_sphere_intersection(ray_origin, ray_direction, sphere_center, sphere_radius, displacement_map):
    """
    Calcula a interseção de um raio com uma esfera e aplica o mapeamento de deslocamento.

    Args:
        ray_origin: Origem do raio (np.array de dimensão 3)
        ray_direction: Direção do raio (np.array de dimensão 3)
        sphere_center: Centro da esfera (np.array de dimensão 3)
        sphere_radius: Raio da esfera (float)
        displacement_map: Mapa de deslocamento (np.array de dimensões (h, w, 2))

    Returns:
        t: Parâmetro de interseção do raio com a esfera (None se não houver interseção)
        displacement: Vetor de deslocamento na interseção (np.array de dimensão 3)
    """
    oc = ray_origin - sphere_center
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - sphere_radius * sphere_radius
    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return None, None
    else:
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        intersection_point = ray_origin + t * ray_direction

        # Obter coordenadas normalizadas no intervalo [0, 1] para o mapeamento de deslocamento
        u = (intersection_point[0] - (sphere_center[0] - sphere_radius)) / (2 * sphere_radius)
        v = (intersection_point[1] - (sphere_center[1] - sphere_radius)) / (2 * sphere_radius)

        # Interpolar o valor de deslocamento do mapa de deslocamento
        displacement_value = np.array([
            np.interp(v, [0, 1], [0, displacement_map.shape[0] - 1]),
            np.interp(u, [0, 1], [0, displacement_map.shape[1] - 1])
        ])
        displacement = displacement_map[int(displacement_value[0]), int(displacement_value[1])]

        return t, displacement

def ray_color(ray_origin, ray_direction, displacement_map):
    """
    Retorna a cor de um raio com mapeamento de deslocamento aplicado.

    Args:
        ray_origin: Origem do raio (np.array de dimensão 3)
        ray_direction: Direção do raio (np.array de dimensão 3)
        displacement_map: Mapa de deslocamento (np.array de dimensões (h, w, 2))

    Returns:
        color: Cor do raio (np.array de dimensão 3)
    """
    sphere_center = np.array([0, 0, -3])  # Posição da esfera
    sphere_radius = 1.0  # Raio da esfera
    t, displacement = ray_sphere_intersection(ray_origin, ray_direction, sphere_center, sphere_radius, displacement_map)

    if t is not None:
        intersection_point = ray_origin + t * ray_direction
        # Aplicar deslocamento em cada coordenada do ponto de interseção
        intersection_point[0] += displacement[0]
        intersection_point[1] += displacement[1]

        # Se houver interseção, retorna vermelho
        return np.array([1, 0, 0])
    else:
        # Se não houver interseção, retorna cor de fundo (preto)
        return np.array([0, 0, 0])


def render(width, height, dm):
    """
    Renderiza a imagem usando ray tracing.
    
    Args:
        width: largura da imagem
        height: altura da imagem
    
    Returns:
        image: imagem renderizada (np.array de dimensões (height, width, 3))
    """
    image = np.zeros((height, width, 3))
    aspect_ratio = width / height
    fov = np.pi / 2  # Campo de visão

    for y in range(height):
        for x in range(width):
            # Normaliza as coordenadas do pixel para o intervalo [-1, 1]
            ray_direction = np.array([
                (2 * (x + 0.5) / width - 1) * aspect_ratio * np.tan(fov / 2),
                (1 - 2 * (y + 0.5) / height) * np.tan(fov / 2),
                -1
            ])
            ray_direction /= np.linalg.norm(ray_direction)  # Normaliza o vetor direção do raio

            color = ray_color(np.array([0, 0, 0]), ray_direction, dm)
            image[y, x] = color

    return image

def main():
    width = 400
    height = 200

    # Caminho para a imagem
    image_path = "disp_mapping_1.jpg"

    # Gerar o mapa de deslocamento
    displacement_map = generate_displacement_map(image_path)

    image = render(width, height, displacement_map)

    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    main()
