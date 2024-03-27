

min_x = -3.8220
max_x = 3.8220
min_z = -32.7600
max_z = -16.5900
y = -3.8416

sub = 6
verts = []
uvs = []
tris = []
for i in range(sub + 1):
    for j in range(sub + 1):
        x = min_x + (max_x - min_x) * i / sub
        z = min_z + (max_z - min_z) * j / sub

        verts.append([x, y, z])

        uvs.append([i / sub, j / sub])

for i in range(sub):
    for j in range (sub):
        tris.append([i * (sub + 1) + j, i * (sub + 1) + j + 1, (i + 1) * (sub + 1) + j])
        tris.append([(i + 1) * (sub + 1) + j + 1, (i + 1) * (sub + 1) + j, i * (sub + 1) + j + 1])




for i in verts:
    print(f'v {i[0]} {i[1]} {i[2]}')
print('')
for i in uvs:
    print(f'uv {i[0]} {i[1]}')
print('')
for i in tris:
    print(f'f {i[0]+1} {i[1]+1} {i[2]+1}')     