#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <cstdint>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include "stb_image_write.h"
#include "stb_image.h"

struct Vec3
{
    float x, y, z;
};

struct Vec2
{
    float x, y;
};

struct StaticMeshVertex
{
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
};

enum class VertexFlags : std::uint32_t
{
    DEFAULT = 0, // vec3 position vec3 normal vec2 uv
    HAS_JOINT_DATA = 1, // uint32 joint indices vec4 joint weights
    HAS_TANGENT_BITANGENT = 2, // vec3 tangent vec3 bitangent
};

struct PhongMaterial
{
    Vec3 diffuse_coefficient;
    Vec3 specular_coefficient;
    float shininess;
    std::string diffuse_map_filename;
    std::string specular_map_filename;
    std::string normal_map_filename;
};

struct StaticMesh
{
    unsigned int indices_begin;
    unsigned int indices_end;
    unsigned int material_index;
};

struct ModelFileContent
{
    struct Header 
    {
        std::uint32_t magic_number = 'hsem';
        std::uint32_t num_meshes;
        std::uint32_t num_vertices;
        std::uint32_t num_indices;
        VertexFlags vertex_flags = VertexFlags::DEFAULT;
        // add padding if needed
    };
    Header header;
    std::vector<StaticMesh> meshes;
    std::vector<StaticMeshVertex> vertices;
    std::vector<unsigned int> indices;
    std::string file_path;
};

struct MaterialFileContent
{
    struct Header
    {
        std::uint32_t magic_number = 'etam';
        // add padding if needed
    };
    Header header;
    PhongMaterial material;
    std::string file_path;
};

void WriteModelFileContent(const ModelFileContent& contents);
void WriteMaterialFileContent(const MaterialFileContent& contents);
void ConvertMesh(const char* path);
void PushChildrenOf(std::vector<aiNode*>& nodes, std::vector<int>& parent, unsigned int begin);
std::pair<std::vector<aiNode*>, std::vector<int>> FlattenNodeHierarchy(aiNode* root);

int main(int argc, char** argv)
{
    for (int i = 1; i < argc; i++)
    {
        ConvertMesh(argv[i]);
    }

    return 0;
}

void ConvertMesh(const char* path)
{
    assert(AI_LMW_MAX_WEIGHTS == 4 && "Expected AI_LMW_MAX_WEIGHTS to be 4.");

    Assimp::Importer importer;
    unsigned int flags = aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_LimitBoneWeights | aiProcess_ValidateDataStructure | aiProcess_GenSmoothNormals |
        aiProcess_CalcTangentSpace;
    const aiScene* scene = importer.ReadFile(path, flags);
    
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        std::cout << "ERROR::ASSIMP::" << importer.GetErrorString() << '\n';
        return;
    }

    std::string path_str = path;
    int last_slash_index = path_str.size() - 1;
    while (last_slash_index >= 0 && path_str[last_slash_index] != '/' && path_str[last_slash_index] != '\\') last_slash_index--;
    std::string file_name = path_str.substr(last_slash_index + 1);
    std::ofstream file(file_name);

    auto flattened_nodes = FlattenNodeHierarchy(scene->mRootNode);
    auto& nodes = flattened_nodes.first;
    auto& parent = flattened_nodes.second;

    ModelFileContent model_file_content;
    model_file_content.header.num_meshes = scene->mNumMeshes;
    model_file_content.file_path = scene->mRootNode->mName.C_Str();
    auto& meshes = model_file_content.meshes;
    auto& indices = model_file_content.indices;
    auto& vertices = model_file_content.vertices;

    // Process meshes
    for (int i = 0; i < scene->mNumMeshes; i++)
    {
        auto mesh = scene->mMeshes[i];

        if (mesh->GetNumUVChannels() > 1)
        {
            std::cout << "Warning: multiple texture coordinates per vertex not supported.\n";
        }

        meshes.emplace_back();
        auto& new_mesh = meshes.back();
        new_mesh.indices_begin = indices.size();
        new_mesh.material_index = mesh->mMaterialIndex;

        unsigned int mesh_index_offset = vertices.size();
        for (int j = 0; j < mesh->mNumFaces; j++)
        {
            auto face = mesh->mFaces[j];
            auto a = face.mIndices[0] + mesh_index_offset;
            auto b = face.mIndices[1] + mesh_index_offset;
            auto c = face.mIndices[2] + mesh_index_offset;
            indices.insert(indices.end(), { a, b, c });
        }

        new_mesh.indices_end = indices.size();

        for (int j = 0; j < mesh->mNumVertices; j++)
        {
            auto vertex = mesh->mVertices[j];
            auto normal = mesh->mNormals[j];
            auto uv = mesh->mTextureCoords[0][j];
            vertices.emplace_back();
            auto& new_vertex = vertices.back();
            new_vertex.position = { vertex.x, vertex.y, vertex.z };
            new_vertex.normal = { normal.x, normal.y, normal.z };
            new_vertex.uv = { uv.x, uv.y };
        }
    }

    model_file_content.header.num_vertices = vertices.size();
    model_file_content.header.num_indices = indices.size();

    // Process materials
    std::vector<MaterialFileContent> material_files_content;
    for (int i = 0; i < scene->mNumMaterials; i++)
    {
        auto material = scene->mMaterials[i];

        material_files_content.emplace_back();
        auto& material_file_content = material_files_content.back();
        if (material->GetName().C_Str() == nullptr) std::cout << "Warning: unnamed material\n";
        material_file_content.file_path = material->GetName().C_Str();

        auto& new_material = material_file_content.material;

        aiColor3D diffuse_color;
        material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse_color);

        aiColor3D specular_color;
        material->Get(AI_MATKEY_COLOR_SPECULAR, specular_color);
        
        material->Get(AI_MATKEY_SHININESS, new_material.shininess);
        

        if (material->GetTextureCount(aiTextureType_DIFFUSE) > 1)
        {
            std::cout << "Warning: Unsupported multiple diffuse textures on material '" << material->GetName().C_Str() << "'\n";
        }

        aiString diffuse_texture_path;
        if (material->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), diffuse_texture_path) != aiReturn_FAILURE)
        {
            if (auto texture = scene->GetEmbeddedTexture(diffuse_texture_path.C_Str()))
            {
                new_material.diffuse_map_filename = texture->mFilename.C_Str();
                auto last_slash_index = new_material.diffuse_map_filename.find_last_of("/");
                assert(last_slash_index != std::string::npos);
                new_material.diffuse_map_filename = new_material.diffuse_map_filename.substr(last_slash_index + 1);

                // Image is compressed (as png etc)
                if (texture->mHeight == 0)
                {
                    std::ofstream ofs(new_material.diffuse_map_filename, std::ios_base::binary);
                    ofs.write((const char*)texture->pcData, texture->mWidth);
                }
                // Image is stored as texel array
                else
                {
                    constexpr int num_channels = 4; // Always true according to Assimp docs (Assimp Data structures)
                    constexpr int texel_size_bytes = 32; // Ditto
                    const int stride_in_bytes = texel_size_bytes * texture->mWidth;
                    stbi_write_png(new_material.diffuse_map_filename.c_str(), texture->mWidth, texture->mHeight, num_channels, texture->pcData, stride_in_bytes);
                }
            }
            else
            {
                new_material.diffuse_map_filename = diffuse_texture_path.C_Str();
            }
        }
        else
        {
            // TODO: handle no diffuse texture
        }

        if (material->GetTextureCount(aiTextureType_SPECULAR) > 1)
        {
            std::cout << "Warning: Unsupported multiple specular textures on material '" << material->GetName().C_Str() << "'\n";
        }

        aiString specular_texture_path;
        if (material->Get(AI_MATKEY_TEXTURE_SPECULAR(0), specular_texture_path) != aiReturn_FAILURE)
        {
            if (auto texture = scene->GetEmbeddedTexture(specular_texture_path.C_Str()))
            {
                new_material.specular_map_filename = texture->mFilename.C_Str();
                auto last_slash_index = new_material.specular_map_filename.find_last_of("/");
                assert(last_slash_index != std::string::npos);
                new_material.specular_map_filename = new_material.specular_map_filename.substr(last_slash_index + 1);

                // Image is compressed (as png etc)
                if (texture->mHeight == 0)
                {   
                    std::ofstream ofs(new_material.specular_map_filename, std::ios_base::binary);
                    ofs.write((const char*)texture->pcData, texture->mWidth);
                }
                // Image is stored as texel array
                else
                {
                    constexpr int num_channels = 4; // Always true according to Assimp docs (Assimp Data structures)
                    constexpr int texel_size_bytes = 32; // Ditto
                    const int stride_in_bytes = texel_size_bytes * texture->mWidth;
                    stbi_write_png(new_material.diffuse_map_filename.c_str(), texture->mWidth, texture->mHeight, num_channels, texture->pcData, stride_in_bytes);
                }
            }
            else
            {
                new_material.specular_map_filename = specular_texture_path.C_Str();
            }
        }
        else 
        {
            // TODO: handling missing specular texture
        }

        aiString normal_texture_path;
        if (material->Get(AI_MATKEY_TEXTURE_NORMALS(0), normal_texture_path) != aiReturn_FAILURE)
        {
            if (auto texture = scene->GetEmbeddedTexture(normal_texture_path.C_Str()))
            {
                new_material.normal_map_filename = texture->mFilename.C_Str();
                auto last_slash_index = new_material.normal_map_filename.find_last_of("/");
                assert(last_slash_index != std::string::npos);
                new_material.normal_map_filename = new_material.normal_map_filename.substr(last_slash_index + 1);

                // Image is compressed (as png etc)
                if (texture->mHeight == 0)
                {
                    std::ofstream ofs(new_material.normal_map_filename, std::ios_base::binary);
                    ofs.write((const char*)texture->pcData, texture->mWidth);
                }
                // Image is stored as texel array
                else
                {
                    constexpr int num_channels = 4; // Always true according to Assimp docs (Assimp Data structures)
                    constexpr int texel_size_bytes = 32; // Ditto
                    const int stride_in_bytes = texel_size_bytes * texture->mWidth;
                    stbi_write_png(new_material.diffuse_map_filename.c_str(), texture->mWidth, texture->mHeight, num_channels, texture->pcData, stride_in_bytes);
                }
            }
            else
            {
                new_material.normal_map_filename = normal_texture_path.C_Str();
            }
        }
        else 
        {
            // TODO: handling missing specular texture
        }
    }

    WriteModelFileContent(model_file_content);

    for (auto& material : material_files_content)
    {
        WriteMaterialFileContent(material);
    }
}

void WriteModelFileContent(const ModelFileContent& content)
{
    assert(content.header.num_meshes == content.meshes.size() && 
           content.header.num_vertices == content.vertices.size() &&
           content.header.num_indices == content.indices.size());

    std::ofstream file(content.file_path, std::ios_base::binary);

    file.write((const char*)&content.header, sizeof(ModelFileContent::Header));
    file.write((const char*)content.meshes.data(), sizeof(content.meshes.front()) * content.meshes.size());
    file.write((const char*)content.vertices.data(), sizeof(content.vertices.front()) * content.vertices.size());
    file.write((const char*)content.indices.data(), sizeof(content.indices.front()) * content.indices.size());
}

void WriteMaterialFileContent(const MaterialFileContent& content)
{
    std::ofstream file(content.file_path, std::ios_base::binary);

    file.write((const char*)&content.material.diffuse_coefficient, sizeof(content.material.diffuse_coefficient));
    file.write((const char*)&content.material.specular_coefficient, sizeof(content.material.specular_coefficient));
    file.write((const char*)&content.material.shininess, sizeof(content.material.shininess));
    auto num_chars_diffuse = content.material.diffuse_map_filename.size();
    file.write((const char*)&num_chars_diffuse, sizeof(num_chars_diffuse));
    file.write(content.material.diffuse_map_filename.c_str(), num_chars_diffuse);
    auto num_chars_specular = content.material.specular_map_filename.size();
    file.write((const char*)&num_chars_specular, sizeof(num_chars_specular));
    file.write(content.material.specular_map_filename.c_str(), num_chars_specular);
    auto num_chars_normal = content.material.normal_map_filename.size();
    file.write((const char*)&num_chars_normal, sizeof(num_chars_normal));
    file.write(content.material.normal_map_filename.c_str(), num_chars_normal);
}

void PushChildrenOf(std::vector<aiNode*>& nodes, std::vector<int>& parent, unsigned int begin)
{
    unsigned int end = nodes.size();
    for (unsigned int i = begin; i < end; i++)
    {
        auto node = nodes[i];
        for (unsigned int j = 0; j < node->mNumChildren; j++)
        {
            nodes.push_back(node->mChildren[j]);
            parent.push_back(i);
        }
    }
    if (end < nodes.size())
    {
        PushChildrenOf(nodes, parent, end);
    }
}

std::pair<std::vector<aiNode*>, std::vector<int>> FlattenNodeHierarchy(aiNode* root)
{
    std::vector<aiNode*> flattened;
    std::vector<int> parent;
    flattened.push_back(root);
    parent.push_back(-1);
    PushChildrenOf(flattened, parent, 0);
    return { flattened, parent };
}