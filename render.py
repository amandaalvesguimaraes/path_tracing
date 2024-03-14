from cmath import atan, sqrt
import cmath
import numpy
from PIL import Image
from multiprocessing import Process, Array
import random, math
import datetime

# classe para passar os dados quando houver algum hit
class rayhit:
    def __init__(self, hitObj,hitPoint, hitNormal, hitDistance, color, ray):
        self.hitObj = hitObj
        self.hitPoint = hitPoint
        self.hitNormal = hitNormal
        self.hitDistance = hitDistance
        self.color = color
        self.ray = ray

# classe principal dos objetos da cena
class scene_object:
    def __init__(self, position = (0,0,0), color = (255,0,0), ka=1, kd=1, ks=1, phongN=1, kr=0, kt=0, refN = 1):
        self.position = position
        self.color = color        
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.phongN = phongN
        self.kr = kr
        self.kt = kt
        self.refN = refN
    
    def getColor(self,p):
        return self.color
    
    def get_secondary_ray_type(self):
        """"
        Pode ser 'kd', 'ks' ou 'kt'
        """
        ktot = self.kd + self.ks + self.kt
        k_rand = random.uniform(0, ktot)
        if k_rand <= self.kd:
            return 'kd'
        elif k_rand <= self.kd + self.ks:
            return 'ks'
        else:
            return 'kt' 
        
    # retorna a normal no ponto p
    def getNormal(self, p):
        return
    
    # retorna 0 se não houver hit, se houver retorna um rayhit
    def intersection(self, origin, direction):
        return

# classe da cena, vai guardar os objetos
class scene_main:
    def __init__(self, lights=[], objs=[], bg_color=(0, 0, 0), ambientLight = (0,0,0)):
        
        # criação de objetos para popular a cena
        self.objs = objs
        self.lights = lights
        self.bg_color = bg_color
        self.ambientLight = ambientLight
    
    
    # DAQUI PRA BAIXO NAO USA MAIS
    def setBackground_Color(self, color):
        self.bg_color = color
    
    def getBackground_Color(self):
        return self.bg_color
    
    def setAmbientLight(self, color):
        self.ambientLight = color
    

# classe das luzes que vao popular scene
class light:
    def __init__(self, position, color):
        self.position = position
        self.color = color

# função principal para criar a imagem test.png com o resultado
def render(res_h, res_v, pxl_size,d,cam_pos,cam_forward,cam_up, scene, max_depth, rays_per_pixel):
    # cria um Image todo preto
    img = Image.new('RGB', (res_h,res_v), color = (0,0,0))
    
    # acha o vetor da camera que aponta para a sua direita
    cam_right = numpy.asarray(numpy.cross(cam_up, cam_forward))
    
    # centro do pixel mais a esquerda e acima
    topleft = cam_pos + cam_forward * d + (cam_up *  (res_v - 1) - cam_right * (res_h - 1)) * pxl_size * 0.5
    
    # quantas threads serão usadas (process)
    thread_count = 12
    
    # o range de cada thread horizontalmente
    xranges = []

    for i in range(thread_count):
        xranges.append(int(i * (res_h / thread_count)))
    xranges.append(res_h)
    
    # lista das threads
    all_threads = []
    # lista das cores de cada thread
    ars = []

    # criacao das threads definindo o espaco horizontal de cada uma
    for t in range(thread_count):
        ars.append(Array("i", range(res_v * (xranges[t + 1] - xranges[t]))))
        ars.append(Array("i", range(res_v * (xranges[t + 1] - xranges[t]))))
        ars.append(Array("i", range(res_v * (xranges[t + 1] - xranges[t]))))

        all_threads.append(Process(target=thread_render, args=(cam_pos,cam_up,cam_right,topleft,pxl_size,scene,xranges[t],xranges[t+1],0,res_v, 
            max_depth, rays_per_pixel, ars[t*3],ars[t*3+1], ars[t*3+2]),daemon=True))
    
    #iniciar as threads
    for x in all_threads:
        x.start()
    
    # esperar todas as threads concluirem
    for x in all_threads:
        x.join()

    # gravacao dos valores calculados pelas threads na imagem 
    for i in range(thread_count):
        for x in range(xranges[i + 1] - xranges[i]):
            for y in range(res_v):
                c = (ars[i * 3 + 0][x * res_v + y] ,ars[i * 3 + 1][x * res_v + y] ,ars[i * 3 + 2][x * res_v + y])              
                img.putpixel((xranges[i] + x, y),c)       

    img.save('test.png')
    print(f'[{datetime.datetime.now().strftime("%H:%M:%S")}] imagem salva')


# funcao que vai obter os dados de pixels em um segmento da imagem renderizada
# os intervalos x0, x1 sao usados usados
def thread_render(cam_pos, cam_up, cam_right, topleft, pxl_size, scene, x0, x1, y0, y1, max_depth, rays_per_pixel, arsR,arsG,arsB):
    
    for x_ in range(x1-x0):
        x = x_ + x0
        
        for y_ in range(y1-y0):
            y = y_ + y0

            ray_dir = normalized((topleft + (cam_up * -y + cam_right * x) * pxl_size) - cam_pos)
            # raycast e feito a partir da camera com a direcao do raio, max_depth e a profundida da recursao(raios secundarios de reflexao e refracao)

            # color = (0,0,0)
            # for j_ in range(rays_per_pixel):
            #     color = colorSum(color, cast(cam_pos,ray_dir, scene, max_depth))
            # color = colorDivide(color, rays_per_pixel)
            c = colorDenormalize(cast_first(cam_pos,ray_dir, scene, max_depth, rays_per_pixel))
            arsR[x_ * y1 + y_] = c[0]
            arsG[x_ * y1 + y_] = c[1]
            arsB[x_ * y1 + y_] = c[2]
    
    print(f'[{datetime.datetime.now().strftime("%H:%M:%S")}] thread end')

def cast_first(origin, direction,scene, counter, rays_per_pixel):
    color = colorNormalize(scene.getBackground_Color())
    hit = trace(origin,direction,scene)
    if hit:
        #for _ in range(rays_per_pixel):
        color = shade_first(hit, scene, counter, rays_per_pixel)
        #print(color)
        
    return (color)

# funcao cast sera chamada recursivamente, a entrada counter definira quantas recursoes serao feitas
def cast(origin, direction,scene, counter):
    color = colorNormalize(scene.getBackground_Color())
    hit = trace(origin,direction,scene)
    if hit:
        color = shade(hit, scene, counter)
        #print(color)
        
    return (color)

def trace(origin, direction, scene:scene_main):
    hit = 0

    # buscamos intersecoes com todos os objetos e retornamos a intersecao mais proxima ao ponto de origem do raio
    for i in range(len(scene.objs)):
        
        hitCheck = scene.objs[i].intersection(origin,direction)

        if hitCheck != 0 and (hit == 0 or hitCheck.hitDistance < hit.hitDistance):
            hit = hitCheck
    
    return hit


def shade_first (hit:rayhit, scene:scene_main, counter, rays_per_pixel):
    # cor do objeto
    color_difuse = colorNormalize(hit.color)
    # cor do pixel iniciada com a luz ambiente
    color =  colorScale(colorMul(color_difuse, colorNormalize(scene.ambientLight)), hit.hitObj.ka)

    #return color_difuse 
    # para cada luz na cena calcular a cor
    for light in scene.lights:
        color_light = colorNormalize(light.color)
        l = light.position - hit.hitPoint
        lDist = numpy.linalg.norm(l)
        l = normalized(l)

        ndotl = numpy.dot(hit.hitNormal, l).real
        
        
        # se recebe luz
        if ndotl > 0:
            shadowHit = trace(hit.hitPoint + l *0.00001, l, scene)
            # print('shadowHit.hitDistance', shadowHit.hitDistance)
            if shadowHit !=0 and shadowHit.hitDistance < lDist:
                continue
            
            # cor difusa
            color = colorSum(color, colorScale(colorMul(color_light, color_difuse), ndotl * hit.hitObj.kd))
            
            rj = 2 * ndotl * hit.hitNormal - l

            view = normalized(-hit.ray)
            rjdotview = numpy.dot(rj,view).real
            if rjdotview < 0:
                rjdotview = 0
            
            # cor especular
            color = colorSum(color, colorScale(color_light , hit.hitObj.ks * numpy.power(rjdotview, hit.hitObj.phongN)))

    # ray recursivo do pathtracing
    if counter > 0:
        sec_color = (0,0,0)
        # tipo ray
        for j in range(rays_per_pixel):
            ray_type = hit.hitObj.get_secondary_ray_type()
            
            if ray_type == 'kd': #difuse
                r1 = random.random()
                r2 = random.random()

                theta = math.acos(math.sqrt(r1))
                phi = 2 * math.pi * r2

                # Convert to Cartesian coordinates
                x = math.sin(theta) * math.cos(phi)
                y = math.sin(theta) * math.sin(phi)
                z = math.cos(theta)

                rayDir = normalized(numpy.array([x, y, z]))

                #view = normalized(hit.ray)
                #rayDir = reflect(view, hit.hitNormal)
                refColor = cast(hit.hitPoint + hit.hitNormal * 0.00001, rayDir, scene, counter-1)
                sec_color = colorSum(sec_color, colorScale(refColor, hit.hitObj.kd))
            elif ray_type == 'ks': #specular
                view = normalized(hit.ray)
                rayDir = reflect(view, hit.hitNormal)
                refColor = cast(hit.hitPoint + rayDir * 0.00001, rayDir, scene, counter-1)
                sec_color = colorSum(sec_color, colorScale(refColor, hit.hitObj.ks))
            elif ray_type == 'kt': #transmission
                pass
        if rays_per_pixel > 0:
            color = colorSum(color, colorDivide(sec_color,rays_per_pixel))
    return color


# funcao shade sera chamada recursivamente, a entrada counter definira quantas recursoes serao feitas
def shade(hit:rayhit, scene:scene_main, counter):
    # cor do objeto
    color_difuse = colorNormalize(hit.color)
    # cor do pixel iniciada com a luz ambiente
    color =  colorScale(colorMul(color_difuse, colorNormalize(scene.ambientLight)), hit.hitObj.ka)

    #return color_difuse 
    # para cada luz na cena calcular a cor
    for light in scene.lights:
        color_light = colorNormalize(light.color)
        l = light.position - hit.hitPoint
        #print(light.position)
        lDist = numpy.linalg.norm(l)
        l = normalized(l)

        ndotl = numpy.dot(hit.hitNormal, l).real
        
        
        # se recebe luz
        if ndotl > 0:
            shadowHit = trace(hit.hitPoint + l *0.00001, l, scene)
            # print('shadowHit.hitDistance', shadowHit.hitDistance)
            if shadowHit !=0 and shadowHit.hitDistance < lDist:
                continue
            
            # cor difusa
            color = colorSum(color, colorScale(colorMul(color_light, color_difuse), ndotl * hit.hitObj.kd))
            
            rj = 2 * ndotl * hit.hitNormal - l

            view = normalized(-hit.ray)
            rjdotview = numpy.dot(rj,view).real
            if rjdotview < 0:
                rjdotview = 0
            
            # cor especular
            color = colorSum(color, colorScale(color_light , hit.hitObj.ks * numpy.power(rjdotview, hit.hitObj.phongN)))

    # ray recursivo do pathtracing
    if counter > 0:
        # tipo ray
        ray_type = hit.hitObj.get_secondary_ray_type()
        
        if ray_type == 'kd': #difuse
            r1 = random.random()
            r2 = random.random()

            theta = math.acos(math.sqrt(r1))
            phi = 2 * math.pi * r2

            # Convert to Cartesian coordinates
            x = math.sin(theta) * math.cos(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(theta)

            rayDir = normalized(numpy.array([x, y, z]))

            #view = normalized(hit.ray)
            #rayDir = reflect(view, hit.hitNormal)
            refColor = cast(hit.hitPoint + hit.hitNormal * 0.00001, rayDir, scene, counter-1)
            color = colorSum(color, colorScale(refColor, hit.hitObj.kd))
        elif ray_type == 'ks': #specular
            view = normalized(hit.ray)
            rayDir = reflect(view, hit.hitNormal)
            refColor = cast(hit.hitPoint + rayDir * 0.00001, rayDir, scene, counter-1)
            color = colorSum(color, colorScale(refColor, hit.hitObj.ks))
        elif ray_type == 'kt': #transmission
            pass

    # contador de rays recursivos
    # if counter > 0:
    #     # refracao
    #     kr = hit.hitObj.kr
    #     if hit.hitObj.kt > 0:
    #         view = normalized(hit.ray)
    #         rayDir = refract(view, normalized(hit.hitNormal), hit.hitObj.refN)
            
    #         if numpy.isscalar(rayDir) == False: # se ha refracao
    #             # cast recursivo da refracao
    #             refColor = cast(hit.hitPoint + rayDir * 0.00001, rayDir, scene, counter-1)
    #             # soma da cor da refracao
    #             color = colorSum(color,colorScale(refColor, hit.hitObj.kt))
    #         else: # se nao ha refracao
    #             kr = 1
        
    #     #reflexao
    #     if kr > 0:
    #         view = normalized(hit.ray)
    #         rayDir = reflect(view, hit.hitNormal)
    #         # cast recursivo da reflexao
    #         refColor = cast(hit.hitPoint + rayDir * 0.00001, rayDir, scene, counter-1)
    #         # soma da cor da reflexao
    #         color = colorSum(color,colorScale(refColor, kr))
    

    return color

# funcao que retorna um vetor normalizado
def normalized(vec):
    n = numpy.linalg.norm(vec)
    if n ==0:
        return vec
    else:
        return vec / n

# função para refletir um raio
def reflect(vec, normal):
    n = normalized(normal)
    return numpy.dot(vec, n) * n * -2 + vec

# refracao: vector = vetor incidencia // normal = normal da superficie // n = n1 / n2 (coeficientes) 
def refract(vec, normal, n):

    w = -vec

    if numpy.dot(w,normal)>0:   # caso entrando no objeto
        ndotw = numpy.dot(normal,w)

        delta = 1 - (1/(n*n)) *(1-ndotw*ndotw)

        if delta < 0:
            return -1
        else:
            t = - (1/n) * w - (numpy.sqrt(delta) - (1/n) * ndotw) * normal
            return t
    else:                       # caso saindo do objeto        
        normal1 = -normal
        ndotw = numpy.dot(normal1,w)
        
        n1 = 1/n

        delta = 1 - (1/(n1*n1)) *(1-ndotw*ndotw)

        if(delta<0):
            return -1
        else:
            t =  - (1/n1) * w - (numpy.sqrt(delta)-(1/n1) * ndotw) * normal1
            return t

# multiplica cores (0-1.0)
def colorMul(color1, color2):
    r1 = color1[0]
    g1 = color1[1]
    b1 = color1[2]
    
    r2 = color2[0]
    g2 = color2[1]
    b2 = color2[2]

    return (r1 * r2, g1 * g2, b1 * b2)

# multiplica cor por um escalar (0-1.0)
def colorScale(color, f):
    return (color[0] * f, color[1] * f, color[2] * f)

# soma duas cores (0-1.0)
def colorSum(color1, color2):
    r1 = color1[0]
    g1 = color1[1]
    b1 = color1[2]
    
    r2 = color2[0]
    g2 = color2[1]
    b2 = color2[2]

    return (r1+r2, g1+g2, b1+b2)


def colorDivide(color, div):
    r = color[0]
    g = color[1]
    b = color[2]

    return (r/div, g/div, b/div)


# passa a cor de (0 - 255) para (0 - 1.0)
def colorNormalize(color):
    return (float(color[0]) / 255.0, float(color[1]) / 255.0, float(color[2]) / 255.0)

# passa a cor de (0 - 1.0) para (0 - 255)
def colorDenormalize(color):
    f = max(1,*color)
    return (int(color[0] * 255.0/f), int(color[1] * 255.0/f), int(color[2] * 255.0/f))

def read_obj_file(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            if parts[0] == 'v':
                # Vertex line
                vertex = [float(p) for p in parts[1:]]
                vertices.append(vertex)
            elif parts[0] == 'f':
                # Face line
                face = [int(p) for p in parts[1:]]
                faces.append(face)

    # print (vertices)
    # print(faces)
    return vertices, faces

class luz_cornell(light):

    def __init__(self, color, vertices, triangles):
        self.vertices = vertices
        self.triangles = triangles

        super().__init__(self.get_center(), color)
    
    def get_center(self):
        center = numpy.array([0,0,0])

        for v in self.vertices:
            center = center + numpy.array(v)
        
        if len(self.vertices) > 0:
            center = center/len(self.vertices)

        return center

def ray_triangle_intersect(orig, dir, v0, v1, v2):
    kEpsilon = 1e-8

    # Compute the plane's normal
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    N = numpy.cross(v0v1, v0v2)  # N
    area2 = numpy.linalg.norm(N)

    # Step 1: finding P

    # Check if the ray and plane are parallel.
    NdotRayDirection = numpy.dot(N, dir)
    if abs(NdotRayDirection) < kEpsilon:  # Almost 0
        return False  # They are parallel, so they don't intersect!

    # Compute d parameter using equation 2
    d = -numpy.dot(N, v0)

    # Compute t (equation 3)
    t = -(numpy.dot(N, orig) + d) / NdotRayDirection

    # Check if the triangle is behind the ray
    if t < 0:
        return False, t  # The triangle is behind

    # Compute the intersection point using equation 1
    P = orig + t * dir

    # Step 2: inside-outside test
    C = numpy.zeros(3)  # Vector perpendicular to triangle's plane

    # Edge 0
    edge0 = v1 - v0
    vp0 = P - v0
    C = numpy.cross(edge0, vp0)
    if numpy.dot(N, C) < 0:
        return False, t  # P is on the right side

    # Edge 1
    edge1 = v2 - v1
    vp1 = P - v1
    C = numpy.cross(edge1, vp1)
    if numpy.dot(N, C) < 0:
        return False, t  # P is on the right side

    # Edge 2
    edge2 = v0 - v2
    vp2 = P - v2
    C = numpy.cross(edge2, vp2)
    if numpy.dot(N, C) < 0:
        return False, t  # P is on the right side

    return True, t  # This ray hits the triangle

class geometry(scene_object):
    def __init__(self, vertices, triangles, color = (255,0,0), ka=1, kd=1, ks=1, phongN=1, kr=0, kt=0, refN = 1):
        #self.position = position
        self.vertices = vertices
        self.triangles = triangles
        # self.color = color        
        # self.ka = ka
        # self.kd = kd
        # self.ks = ks
        # self.phongN = phongN
        # self.kr = kr
        # self.kt = kt
        # self.refN = refN        
        super().__init__((0,0,0), color, ka, kd, ks, phongN, kr, kt, refN)
    
    def getNormal(self, p):
        return self.normal
    
    def getColor(self, p):
        return super().getColor(p)

    def intersection(self, origin, direction):

        #formula para interseção demonstrada no scratchapixel.com
        max_distance = 999999999999999
        hit = 0
        for t in self.triangles:
            v0 = numpy.array(self.vertices[t[0] - 1])
            v1 = numpy.array(self.vertices[t[1] - 1])
            v2 = numpy.array(self.vertices[t[2] - 1])
            
            hit_ocurred, t = ray_triangle_intersect(origin, direction, v0, v1, v2)

            if hit_ocurred:
                if t < max_distance:
                    max_distance = t
                    normal = normalized(numpy.cross(v0 - v1, v0 - v2))
                    hit = rayhit(self, origin + t * direction, normal, t, self.color, t * direction)
                
        return hit

def read_sdl_file(file_path):
    objs = []
    lights = []
    eye = (0.0, 0.0, 5.7)
    # size 200 200
    # ortho -1 -1 1 1
    # background 0.0 0.0 0.0
    # ambient 0.5

    background = (0,0,0)
    ambient = colorDenormalize((0.5, 0.5, 0.5))

    # npaths 10
    # tonemapping 1.0
    # seed 9

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            if parts[0] == 'object':
                
                v, f = read_obj_file(f'objects/{parts[1]}')
                rgb = colorDenormalize((float(parts[2]), float(parts[3]), float(parts[4])))
                ka = float(parts[5])
                kd = float(parts[6])
                ks = float(parts[7])
                kt = float(parts[8])
                n = float(parts[9]) ## TODO: esse n é o que?
                geo = geometry(vertices=v, triangles=f, color=rgb,ka=ka,kd=kd,ks=ks,kt=kt)

                objs.append(geo)
            if parts[0] == 'light':
                # print('light')
                # print(parts)
                v, f = read_obj_file(f'objects/{parts[1]}')
                rgb = colorDenormalize((float(parts[2]), float(parts[3]), float(parts[4])))

                #new_light = luz_cornell(position=[0,0,0], color=rgb,vertices=v,triangles=f)
                new_light = luz_cornell(rgb, v, f)
                lights.append(new_light)
            if parts[0] == 'background':
                background = colorDenormalize((float(parts[1]), float(parts[2]), float(parts[3])))
            if parts[0] == 'ambient':
                ambient = colorDenormalize((float(parts[1]), float(parts[1]), float(parts[1])))
            if parts[0] == 'eye':
                eye = colorDenormalize((float(parts[1]), float(parts[2]), float(parts[3])))

    new_scene = scene_main(lights=lights, objs=objs, bg_color=background, ambientLight=ambient)   
                
    return new_scene


if __name__ == '__main__' :
    
    # nova cena e criada que guardara os objetos e luzes
    new_scene = read_sdl_file('objects/cornellroom2/cornellroom2.sdl') # scene_main()

    # multiplicador das coordenadas, para ajustar as entradas ao espaco
    xyz_coord = numpy.array([1, 1, 1])

    for obj in new_scene.objs:
        for i in range(len(obj.vertices)):
            # print(obj.vertices[i])
            obj.vertices[i] = obj.vertices[i] * xyz_coord
            # print(obj.vertices[i])

    cam_forward = normalized(numpy.array([0,0,-1]))
    cam_up = normalized(numpy.array([0,1,0]))
    cam_pos = numpy.array([0,0,15.7])
    res_horizontal = 200
    res_vertical = 200
    max_depth = 1
    size_pixel = 0.05
    cam_dist = 40
    rays_per_pixel = 100

    # checa se cam_forward e cam_up são aceitos
    if (cam_forward[0] == 0 and cam_forward[1] == 0 and cam_forward[2] == 0) or (cam_up[0] == 0 and cam_up[1] == 0 and cam_up[2] == 0):
        print('cam_forward e cam_up não podem ser [0,0,0] ou paralelas')
    else: # Render da imagem
        print(f'[{datetime.datetime.now().strftime("%H:%M:%S")}] gerando imagem...')
        render(res_horizontal, res_vertical, size_pixel,cam_dist, cam_pos, cam_forward, cam_up, new_scene, max_depth, rays_per_pixel)