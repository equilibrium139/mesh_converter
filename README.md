# mesh_converter
Convert 3d model data from formats supported by Assimp library (FBX, OBJ, GLTF, etc) to a format which can be read and directly sent to the GPU. 
This project was created for a couple reasons:
1. Gain experience with creating custom file formats
2. Remove the Assimp dependency from my other projects
3. Cut the large startup time for Assimp to load large models

Converting all models to GLTF format and loading GLTF would have likely been much simpler but I enjoyed the learning experience (and will enjoy it even more as I fix
inevitable bugs).

## Model file format

Model files have the .model extension. This file contains all the model's vertices and indices (faces) in one buffer each. The file also contains meshes which are
simply indices into the index buffer. The meshes also contain an index to the material they use. The materials are stored inside the model file, although they might
later be stored in their own files. 

1. header- magic number, num meshes, num vertices, num indices, num materials
2. meshes- begin index into indices, end index into indices, material index
3. vertices
4. indices
5. materials- diffuse coefficient, specular coefficient, diffuse map filename, specular map filename, normal map filename

## Skeleton file format

This file format uses the .skeleton extension. The skeleton is stored as a flat array of joints. Each joint contains a 3x4 matrix which transforms the joint's 
vertices from joint space to model space. This is also known as the inverse bind pose matrix. Each joint also contains an index to its parent. A joint's children are 
always at greater indices in the array.

## Animation clip file format

This file format uses the .animation extension. For now no compression is being done as it's not really needed yet. This file contains an array of skeleton poses. A skeleton pose,
as its name suggests, mathematically describes the pose of the skeleton at a certain frame of the animation. Each skeleton pose is an array of joint poses, which are represented
in SRT (vec3 scale, quat rotation, vec3 translation) format. The number of these joint poses matches the number of joints in the skeleton.

1. header- u32 magic number, u32 frame count, f32 frames per second, bool32 loops
2. skeleton poses