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
later be storede in their own files.

1. header
2. meshes
3. vertices
4. indices
5. materials

## Material file format

Currently materials are stored in their own file although I'm leaning towards storing them inside the model to have less files to deal with. Only Phong materials are
supported for now.

1. header
2. diffuse coefficient
3. specular coefficient
4. diffuse map filename
5. specular map filename
6. normal map filename

## Skeleton file format

This file format uses the .skeleton extension. The skeleton is stored as a flat array of joints. Each joint contains a 3x4 matrix which transforms the joint's 
vertices from joint space to model space.
