#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <cstdint>
#include <cstring>
#include <memory>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include "stb_image_write.h"
#include "stb_image.h"

struct Vec4
{
    float x, y, z, w;
    float& operator[](int i)
    {
        assert(i >= 0 && i < 4);
        auto component_address = &x + i;
        return *component_address;
    }
};

struct Quat
{
    float w, x, y, z;
};

struct Vec3
{
    float x, y, z;
};

struct Vec2
{
    float x, y;
};

struct Mat3x4
{
    float column1[3];
    float column2[3];
    float column3[3];
    float column4[3];
};

// Be very very careful changing the ordering of members. Lots of byte fuckery and assumptions going on below
struct Vertex
{
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
    Vec3 tangent;
    std::uint32_t joint_indices;
    Vec4 joint_weights;    
};

enum class VertexFlags : std::uint32_t
{
    DEFAULT = 0, // vec3 position vec3 normal vec2 uv
    HAS_TANGENT = 1 << 0, // vec3 tangent
    HAS_JOINT_DATA = 1 << 1, // uint32 joint indices vec4 joint weights
};

inline VertexFlags operator | (VertexFlags lhs, VertexFlags rhs)
{
    using T = std::underlying_type_t<VertexFlags>;
    return (VertexFlags)((T)lhs | (T)rhs);
}

inline VertexFlags& operator |= (VertexFlags& lhs, VertexFlags rhs)
{
    lhs = lhs | rhs;
    return lhs;
}

inline VertexFlags operator & (VertexFlags lhs, VertexFlags rhs)
{
    using T = std::underlying_type_t<VertexFlags>;
    return (VertexFlags)((T)lhs & (T)rhs);
}

inline VertexFlags& operator &= (VertexFlags& lhs, VertexFlags rhs)
{
    lhs = lhs & rhs;
    return lhs;
}

inline bool HasFlag(VertexFlags flags, VertexFlags flag_to_check)
{
    return (std::underlying_type_t<VertexFlags>)(flags & flag_to_check) != 0;
}

enum class PhongMaterialFlags : std::uint32_t
{
    DEFAULT = 0,
    DIFFUSE_WITH_ALPHA = 1,
};

inline PhongMaterialFlags operator | (PhongMaterialFlags lhs, PhongMaterialFlags rhs)
{
    using T = std::underlying_type_t<PhongMaterialFlags>;
    return (PhongMaterialFlags)((T)lhs | (T)rhs);
}

inline PhongMaterialFlags& operator |= (PhongMaterialFlags& lhs, PhongMaterialFlags rhs)
{
    lhs = lhs | rhs;
    return lhs;
}

inline PhongMaterialFlags operator & (PhongMaterialFlags lhs, PhongMaterialFlags rhs)
{
    using T = std::underlying_type_t<PhongMaterialFlags>;
    return (PhongMaterialFlags)((T)lhs & (T)rhs);
}

inline PhongMaterialFlags& operator &= (PhongMaterialFlags& lhs, PhongMaterialFlags rhs)
{
    lhs = lhs & rhs;
    return lhs;
}


inline bool HasFlag(PhongMaterialFlags flags, PhongMaterialFlags flag_to_check)
{
    return (std::underlying_type_t<PhongMaterialFlags>)(flags & flag_to_check) != 0;
}

struct PhongMaterial
{
    Vec3 diffuse_coefficient;
    Vec3 specular_coefficient;
    float shininess;
    PhongMaterialFlags flags = PhongMaterialFlags::DEFAULT;
    std::string diffuse_map_filename;
    std::string specular_map_filename;
    std::string normal_map_filename;
};

struct Mesh
{
    unsigned int indices_begin;
    unsigned int indices_end;
    unsigned int material_index;
};

struct ModelFile
{
    static constexpr std::uint32_t correct_magic_number = 'ldom';
    struct Header 
    {
        std::uint32_t magic_number = correct_magic_number;
        std::uint32_t num_meshes;
        std::uint32_t num_vertices;
        std::uint32_t num_indices;
        std::uint32_t num_materials;
        VertexFlags vertex_flags = VertexFlags::DEFAULT;
        // add padding if needed
    };
    Header header;
    std::vector<Mesh> meshes;
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<PhongMaterial> materials;
    std::string name;
};

struct Joint
{
    Mat3x4 model_to_joint_transform;
    int parent;
};

struct SkeletonFile
{
    static constexpr std::uint32_t correct_magic_number = 'ntks';
    struct Header
    {
        std::uint32_t magic_number = correct_magic_number;
        std::uint32_t num_joints;
    };
    Header header;
    std::vector<Joint> joints;
    std::vector<std::string> joint_names;
};

struct JointPose
{
    Quat rotation;
    Vec3 translation;
    Vec3 scale;
};

struct SkeletonPose
{
    std::vector<JointPose> joint_poses;
};

struct AnimationClipFile
{
    static constexpr std::uint32_t correct_magic_number = 'pilc';
    struct Header
    {
        using bool32 = std::uint32_t; // for padding purposes
        std::uint32_t magic_number = correct_magic_number;
        std::uint32_t frame_count;
        float frames_per_second; 
        bool32 loops = false; // fix later
        // add padding if needed
    };
    Header header;
    std::vector<SkeletonPose> skeleton_poses; // number of poses = frame_count + 1 or frame_count if loops
    std::string name;
};

struct AssimpNode
{
    const aiNode* node;
    int parent;
};

SkeletonFile ExtractSkeleton(const std::vector<AssimpNode>& nodes, const std::vector<aiMesh*>& meshes);
std::vector<AnimationClipFile> ExtractAnimationClips(const std::vector<aiAnimation*>& assimp_animations, const std::vector<AssimpNode>& nodes, const SkeletonFile& skeleton);
ModelFile ExtractModels(const std::vector<aiMesh*>& assimp_meshes, const std::string& file_path, const SkeletonFile& skeleton);
std::vector<PhongMaterial> ExtractMaterials(const std::vector<aiMaterial*>& assimp_materials, const aiScene* scene);
void WriteSkeletonFile(const SkeletonFile& skeleton, const std::string& file_path);
void WriteAnimationFiles(const AnimationClipFile& animation);
void WriteModelFile(const ModelFile& model);
void ConvertMesh(const char* path);
void PushChildrenOf(std::vector<AssimpNode>& nodes, unsigned int begin);
std::vector<AssimpNode> FlattenNodeHierarchy(aiNode* root);
Mat3x4 ToGLMMat(const aiMatrix4x4& assimp_matrix);
SkeletonFile LoadSkeleton(const std::string_view path);

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: mesh_converter model1_path [model2_path] ... [modeln_path]\n";
    }

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
        aiProcess_CalcTangentSpace | aiProcess_JoinIdenticalVertices;
    const aiScene* scene = importer.ReadFile(path, flags);
    
    if (!scene /*|| scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE*/ || !scene->mRootNode)
    {
        std::cout << "ERROR::ASSIMP::" << importer.GetErrorString() << '\n';
        return;
    }

    std::vector<aiAnimation*> skeleton_animations(scene->mAnimations, scene->mAnimations + scene->mNumAnimations);
    std::vector<aiMesh*> model_meshes(scene->mMeshes, scene->mMeshes + scene->mNumMeshes);
    std::vector<aiMaterial*> model_materials(scene->mMaterials, scene->mMaterials + scene->mNumMaterials);

    auto nodes = FlattenNodeHierarchy(scene->mRootNode);

    // This flag really just means that there are no meshes in the scene.
    // Can use this to process animation data 
    if (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)
    {
        assert(scene->mNumAnimations == 1); 

        std::cout << "Enter skeleton file path: ";
        std::string skeleton_file_path;
        std::cin >> skeleton_file_path;

        auto skeleton = LoadSkeleton(skeleton_file_path);
        auto animation_clip = ExtractAnimationClips(skeleton_animations, nodes, skeleton);
        assert(animation_clip.size() == 1);
        WriteAnimationFiles(animation_clip[0]);
        return;
    }

    std::string model_name;
    std::cout << "Enter model name: ";
    std::cin >> model_name;

    auto skeleton_file = ExtractSkeleton(nodes, model_meshes);
    auto animation_clip_files = ExtractAnimationClips(skeleton_animations, nodes, skeleton_file);
    auto model_file = ExtractModels(model_meshes, model_name + ".model", skeleton_file);
    model_file.materials = ExtractMaterials(model_materials, scene);
    model_file.header.num_materials = (std::uint32_t)model_file.materials.size();

    if (skeleton_file.header.num_joints > 0)
    {
        WriteSkeletonFile(skeleton_file, model_name + ".skeleton");

        model_file.header.vertex_flags |= VertexFlags::HAS_JOINT_DATA;

        for (auto& animation_clip : animation_clip_files)
        {
            WriteAnimationFiles(animation_clip);
        }
    }

    const bool model_has_normal_map = model_file.materials[0].normal_map_filename.size() > 0;
    if (model_has_normal_map) model_file.header.vertex_flags |= VertexFlags::HAS_TANGENT;

    WriteModelFile(model_file);
}

SkeletonFile ExtractSkeleton(const std::vector<AssimpNode>& nodes, const std::vector<aiMesh*>& meshes)
{
    const auto num_nodes = nodes.size();

    // Mark all nodes which correspond to a bone
    std::vector<aiBone*> node_bones(num_nodes, nullptr);
    const auto num_meshes = meshes.size();
    for (auto i = 0u; i < num_meshes; i++)
    {
        auto mesh = meshes[i];

        const auto num_bones = mesh->mNumBones;
        for (auto j = 0u; j < num_bones; j++)
        {
            auto bone = mesh->mBones[j];

            for (auto k = 0u; k < num_nodes; k++)
            {
                if (bone->mName == nodes[k].node->mName)
                {
                    node_bones[k] = bone;
                    break;
                }
            }
        }
    }

    SkeletonFile skeleton_file;
    auto& joints = skeleton_file.joints;
    auto& joint_names = skeleton_file.joint_names;

    // Add only nodes which correspond to a bone to the skeleton
    for (auto i = 0u; i < num_nodes; i++)
    {
        auto bone = node_bones[i];

        if (bone)
        {
            joints.emplace_back();
            Joint& joint = joints.back();
            joint_names.emplace_back(bone->mName.C_Str());
            joint.model_to_joint_transform = ToGLMMat(bone->mOffsetMatrix);

            auto node_parent_index = nodes[i].parent;
            if (node_parent_index >= 0 && node_bones[node_parent_index] != nullptr)
            {
                auto parent_iter = std::find(joint_names.begin(), joint_names.end(), nodes[node_parent_index].node->mName.C_Str());
                assert(parent_iter != joint_names.end());
                joint.parent = (int)std::distance(joint_names.begin(), parent_iter);
            }
            else joint.parent = -1;
        }
    }
    
    skeleton_file.header.num_joints = (std::uint32_t)joints.size();
    return skeleton_file;
}

ModelFile ExtractModels(const std::vector<aiMesh*>& assimp_meshes, const std::string& file_path, const SkeletonFile& skeleton)
{
    ModelFile model_file;
    model_file.header.num_meshes = (unsigned int)assimp_meshes.size();
    model_file.name = file_path;
    auto& meshes = model_file.meshes;
    auto& indices = model_file.indices;
    auto& vertices = model_file.vertices;

    for (auto i = 0u; i < assimp_meshes.size(); i++)
    {
        auto assimp_mesh = assimp_meshes[i];

        if (assimp_mesh->GetNumUVChannels() > 1)
        {
            std::cout << "Warning: multiple texture coordinates per vertex not supported.\n";
        }

        meshes.emplace_back();
        auto& new_mesh = meshes.back();
        new_mesh.indices_begin = (unsigned int)indices.size();
        new_mesh.material_index = assimp_mesh->mMaterialIndex;

        auto mesh_index_offset = (unsigned int)vertices.size();
        for (auto j = 0u; j < assimp_mesh->mNumFaces; j++)
        {
            auto face = assimp_mesh->mFaces[j];
            auto a = face.mIndices[0] + mesh_index_offset;
            auto b = face.mIndices[1] + mesh_index_offset;
            auto c = face.mIndices[2] + mesh_index_offset;
            indices.insert(indices.end(), { a, b, c });
        }

        new_mesh.indices_end = (unsigned int)indices.size();
        const auto mesh_vertices_begin = vertices.size();
        const auto num_mesh_vertices = assimp_mesh->mNumVertices;

        for (auto j = 0u; j < num_mesh_vertices; j++)
        {
            auto vertex = assimp_mesh->mVertices[j];
            auto normal = assimp_mesh->mNormals[j];
            auto uv = assimp_mesh->mTextureCoords[0][j];
            auto tangent = assimp_mesh->mTangents[j];
            vertices.emplace_back();
            auto& new_vertex = vertices.back();
            new_vertex.position = { vertex.x, vertex.y, vertex.z };
            new_vertex.normal = { normal.x, normal.y, normal.z };
            new_vertex.uv = { uv.x, uv.y };
            new_vertex.tangent = { tangent.x, tangent.y, tangent.z };
            new_vertex.joint_indices = 0;
            new_vertex.joint_weights = { 0.0f, 0.0f, 0.0f, 0.0f };
        }

        std::vector<int> num_joints_affecting(num_mesh_vertices, 0);

        for (auto j = 0u; j < assimp_mesh->mNumBones; j++)
        {
            auto bone = assimp_mesh->mBones[j];
            std::uint32_t joint_index = 0;
            while (joint_index < skeleton.header.num_joints && bone->mName.C_Str() != skeleton.joint_names[joint_index]) joint_index++;
            assert(joint_index < skeleton.header.num_joints);

            const auto num_vertices_affected = bone->mNumWeights;
            for (auto k = 0u; k < num_vertices_affected; k++)
            {
                auto mesh_vertex_index = bone->mWeights[k].mVertexId;
                auto weight = bone->mWeights[k].mWeight;
                auto model_vertex_index = mesh_vertex_index + mesh_vertices_begin;
                auto& vertex = vertices[model_vertex_index];
                auto& num_joints_affecting_vertex = num_joints_affecting[mesh_vertex_index];
                vertex.joint_indices = (joint_index << (8 * num_joints_affecting_vertex)) | vertex.joint_indices;
                vertex.joint_weights[num_joints_affecting_vertex] = weight;
                num_joints_affecting_vertex++;
            }
        }
    }

    model_file.header.num_vertices = (unsigned int)vertices.size();
    model_file.header.num_indices = (unsigned int)indices.size();

    return model_file;
}

std::vector<PhongMaterial> ExtractMaterials(const std::vector<aiMaterial*>& assimp_materials, const aiScene* scene)
{
    static std::vector<aiTextureType> unsupported_texture_types =
    {
        {aiTextureType_NONE},
        {aiTextureType_AMBIENT},
        {aiTextureType_EMISSIVE},
        {aiTextureType_HEIGHT},
        {aiTextureType_SHININESS},
        {aiTextureType_DISPLACEMENT},
        {aiTextureType_LIGHTMAP},
        {aiTextureType_REFLECTION},
        {aiTextureType_BASE_COLOR},
        {aiTextureType_NORMAL_CAMERA},
        {aiTextureType_EMISSION_COLOR},
        {aiTextureType_METALNESS},
        {aiTextureType_DIFFUSE_ROUGHNESS},
        {aiTextureType_AMBIENT_OCCLUSION},
        {aiTextureType_SHEEN},
        {aiTextureType_CLEARCOAT},
        {aiTextureType_TRANSMISSION},
        {aiTextureType_UNKNOWN}
    };

    std::vector<PhongMaterial> materials;

    for (auto i = 0u; i < assimp_materials.size(); i++)
    {
        auto material = assimp_materials[i];

        std::cout << material->GetName().C_Str() << '\n';

        for (auto type : unsupported_texture_types)
        {
            if (material->GetTextureCount(type) > 0)
            {
                std::cout << "Warning: unsupported texture type '" << TextureTypeToString(type) << "' on material '" << material->GetName().C_Str() << "'\n";
            }
        }

        materials.emplace_back();
        auto& new_material = materials.back();

        aiColor3D diffuse_color;
        material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse_color);
        new_material.diffuse_coefficient = { diffuse_color.r, diffuse_color.g, diffuse_color.b };

        aiColor3D specular_color;
        material->Get(AI_MATKEY_COLOR_SPECULAR, specular_color);
        new_material.specular_coefficient = { specular_color.r, specular_color.g, specular_color.b };

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
                    std::ofstream ofs(new_material.diffuse_map_filename, std::ios::binary);
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

        if (material->GetTextureCount(aiTextureType_OPACITY) > 1)
        {
            std::cout << "Warning: Unsupported multiple specular textures on material '" << material->GetName().C_Str() << "'\n";
        }

        aiString opacity_texture_path;
        if (material->Get(AI_MATKEY_TEXTURE_OPACITY(0), opacity_texture_path) != aiReturn_FAILURE)
        {
            if (opacity_texture_path == diffuse_texture_path)
            {
                new_material.flags |= PhongMaterialFlags::DIFFUSE_WITH_ALPHA;
            }
            else
            {
                std::cout << "Warning: Opacity map texture only supported if embedded in diffuse texture\n";
            }
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
                    std::ofstream ofs(new_material.specular_map_filename, std::ios::binary);
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
                    std::ofstream ofs(new_material.normal_map_filename, std::ios::binary);
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

    return materials;
}

static JointPose ToJointPose(const aiMatrix4x4 matrix)
{
    aiQuaternion quat;
    aiVector3D trans;
    aiVector3D scale;
    matrix.Decompose(scale, quat, trans);
    JointPose joint_pose;
    joint_pose.rotation = { quat.w, quat.x, quat.y, quat.z };
    joint_pose.translation = { trans.x, trans.y, trans.z };
    joint_pose.scale = { scale.x, scale.y, scale.z };
    return joint_pose;
}

static Vec3 Interpolate(const Vec3& a, const Vec3& b, float t)
{
    Vec3 interpolated = {
        .x = (1 - t) * a.x + t * b.x,
        .y = (1 - t) * a.y + t * b.y,
        .z = (1 - t) * a.z + t * b.z
    };
}

static JointPose ToJointPose(const aiVector3D& scaling, const aiQuaternion& rotation, const aiVector3D& translation)
{
    JointPose joint_pose;
    joint_pose.scale = { scaling.x, scaling.y, scaling.z };
    joint_pose.rotation = { rotation.w, rotation.x, rotation.y, rotation.z };
    joint_pose.translation = { translation.x, translation.y, translation.z };
    return joint_pose;
}

static JointPose SampleAt(aiNodeAnim* anim, float tick)
{
    static Assimp::Interpolator<aiVector3D> lerp;
    static Assimp::Interpolator<aiQuaternion> slerp;

    for (int i = 0; i < anim->mNumPositionKeys; i++)
    {
        if (anim->mPositionKeys[i].mTime > tick)
        {
            // Nothing to interpolate between
            if (i == 0)
            {
                return ToJointPose(anim->mScalingKeys[0].mValue, anim->mRotationKeys[0].mValue, anim->mPositionKeys[0].mValue);
            }
            else
            {
                float t = (tick - anim->mPositionKeys[i - 1].mTime) / (anim->mPositionKeys[i].mTime - anim->mPositionKeys[i - 1].mTime);
                aiVector3D interpolated_position, interpolated_scale;
                aiQuaternion interpolated_rotation;
                lerp(interpolated_position, anim->mPositionKeys[i - 1].mValue, anim->mPositionKeys[i].mValue, t);
                lerp(interpolated_scale, anim->mScalingKeys[i - 1].mValue, anim->mScalingKeys[i].mValue, t);
                slerp(interpolated_rotation, anim->mRotationKeys[i - 1].mValue, anim->mRotationKeys[i].mValue, t);
                return ToJointPose(interpolated_scale, interpolated_rotation, interpolated_position);
            }
        }
        if (i == anim->mNumPositionKeys - 1)
        {
            // Also nothing to interpolate between
            return ToJointPose(anim->mScalingKeys[anim->mNumPositionKeys - 1].mValue, anim->mRotationKeys[anim->mNumPositionKeys - 1].mValue, anim->mPositionKeys[anim->mNumPositionKeys - 1].mValue);
        }
    }
}

std::vector<AnimationClipFile> ExtractAnimationClips(const std::vector<aiAnimation*>& assimp_animations, const std::vector<AssimpNode>& nodes, const SkeletonFile& skeleton)
{
    std::vector<AnimationClipFile> animation_clip_files;

    /*const aiMatrix4x4 identity_matrix;
    const JointPose identity_joint_pose = ToJointPose(identity_matrix);
    SkeletonPose identity_skeleton_pose;
    identity_skeleton_pose.joint_poses.resize(skeleton.header.num_joints);
    for (auto& joint_pose : identity_skeleton_pose.joint_poses) joint_pose = identity_joint_pose;*/

    SkeletonPose original_pose;

    original_pose.joint_poses.resize(skeleton.header.num_joints);
    for (int i = 0; i < skeleton.header.num_joints; i++)
    {
        auto& joint_name = skeleton.joint_names[i];

        int node_index = 0;
        while (node_index < nodes.size() && joint_name != nodes[node_index].node->mName.C_Str()) node_index++;
        assert(node_index < nodes.size());
        auto node = nodes[node_index].node;

        original_pose.joint_poses[i] = ToJointPose(node->mTransformation);
    }

    for (auto& assimp_animation : assimp_animations)
    {
        // assert(assimp_animation->mNumChannels == skeleton.header.num_joints);
        animation_clip_files.emplace_back();
        auto& animation_clip = animation_clip_files.back();
        // Not sure why Assimp is storing the duration in ticks as a double but I'm going to assume that it's
        // an integer number
        assert(std::abs(assimp_animation->mDuration - std::floor(assimp_animation->mDuration)) < 0.01 || 
               std::abs(assimp_animation->mDuration - std::ceil(assimp_animation->mDuration)) < 0.01);
        const float duration_in_seconds = assimp_animation->mDuration / assimp_animation->mTicksPerSecond;
        animation_clip.header.frames_per_second = 30; 
        animation_clip.header.frame_count = animation_clip.header.frames_per_second * duration_in_seconds;
        //animation_clip.header.frame_count = (std::uint32_t)assimp_animation->mDuration;
        //animation_clip.header.frames_per_second = (float)assimp_animation->mTicksPerSecond;
        animation_clip.name = assimp_animation->mName.C_Str();
        auto& skeleton_poses = animation_clip.skeleton_poses;
        const auto pose_count = animation_clip.header.frame_count + (animation_clip.header.loops ? 0 : 1);
        skeleton_poses.resize(pose_count);
        for (auto& pose : skeleton_poses) pose = original_pose;

        const float tick_delta = assimp_animation->mTicksPerSecond / animation_clip.header.frames_per_second;
        for (auto i = 0u; i < assimp_animation->mNumChannels; i++)
        {
            auto node_anim = assimp_animation->mChannels[i];

            auto node_joint_index = 0u;
            while (node_joint_index < skeleton.header.num_joints && skeleton.joint_names[node_joint_index] != node_anim->mNodeName.C_Str()) node_joint_index++;
            assert(node_joint_index < skeleton.header.num_joints);

            assert(node_anim->mNumPositionKeys == node_anim->mNumRotationKeys && node_anim->mNumPositionKeys == node_anim->mNumScalingKeys);
            //assert(node_anim->mNumPositionKeys >= pose_count || node_anim->mNumPositionKeys == 1);

            float tick = 0.0f;
            for (auto& skeleton_pose : skeleton_poses)
            {
                skeleton_pose.joint_poses[node_joint_index] = SampleAt(node_anim, tick);
                tick += tick_delta;
            }

            // joint pose is constant over the animation
           /* if (node_anim->mNumPositionKeys == 1)
            {
                auto scaling = node_anim->mScalingKeys[0].mValue;
                auto rotation = node_anim->mRotationKeys[0].mValue;
                auto translation = node_anim->mPositionKeys[0].mValue;
                JointPose joint_pose;
                joint_pose.scale = { scaling.x, scaling.y, scaling.z };
                joint_pose.rotation = { rotation.w, rotation.x, rotation.y, rotation.z };
                joint_pose.translation = { translation.x, translation.y, translation.z };
                for (auto& pose : skeleton_poses)
                {
                    pose.joint_poses[node_joint_index] = joint_pose;
                }
            }
            else if (node_anim->mNumPositionKeys != pose_count)
            {
                std::cout << "Stuff\n";
            }
            else
            {
                for (auto j = 0u; j < pose_count; j++)
                {
                    auto scaling = node_anim->mScalingKeys[j].mValue;
                    auto rotation = node_anim->mRotationKeys[j].mValue;
                    auto translation = node_anim->mPositionKeys[j].mValue;
                    JointPose current_frame_joint_pose;
                    current_frame_joint_pose.scale = { scaling.x, scaling.y, scaling.z };
                    current_frame_joint_pose.rotation = { rotation.w, rotation.x, rotation.y, rotation.z };
                    current_frame_joint_pose.translation = { translation.x, translation.y, translation.z };

                    auto& current_frame_skeleton_pose = skeleton_poses[j];
                    current_frame_skeleton_pose.joint_poses[node_joint_index] = current_frame_joint_pose;
                }
            }*/
        }
    }

    return animation_clip_files;
}

void WriteSkeletonFile(const SkeletonFile& skeleton, const std::string& file_path)
{
    std::ofstream file(file_path, std::ios::binary);
    
    file.write((const char*)&skeleton.header, sizeof(SkeletonFile::Header));
    file.write((const char*)skeleton.joints.data(), skeleton.joints.size() * sizeof(Joint));
    for (auto& name : skeleton.joint_names)
    {
        file.write(name.c_str(), name.size() + 1);
    }
}

void WriteAnimationFiles(const AnimationClipFile& animation_clip)
{
    const auto pose_count = animation_clip.header.frame_count + (animation_clip.header.loops ? 0 : 1);
    assert(pose_count == animation_clip.skeleton_poses.size());

    std::ofstream file(animation_clip.name + ".animation", std::ios::binary);

    file.write((const char*)&animation_clip.header, sizeof(AnimationClipFile::Header));
    
    for (auto& pose : animation_clip.skeleton_poses)
    {
        file.write((const char*)pose.joint_poses.data(), pose.joint_poses.size() * sizeof(pose.joint_poses.front()));
    }
}

void WriteModelFile(const ModelFile& model)
{
    assert(model.header.num_meshes == model.meshes.size() && 
           model.header.num_vertices == model.vertices.size() &&
           model.header.num_indices == model.indices.size());

    std::ofstream file(model.name, std::ios::binary);

    file.write((const char*)&model.header, sizeof(ModelFile::Header));
    file.write((const char*)model.meshes.data(), sizeof(model.meshes.front()) * model.meshes.size());
    
    const bool needs_tangents = HasFlag(model.header.vertex_flags, VertexFlags::HAS_TANGENT);
    const bool needs_joint_data = HasFlag(model.header.vertex_flags, VertexFlags::HAS_JOINT_DATA);

    // nothing to do, write full vertex buffer as is
    if (needs_tangents && needs_joint_data) file.write((const char*)model.vertices.data(), sizeof(model.vertices.front()) * model.vertices.size());
    else
    {
        // Copy only needed vertex data to the file
        
        //                                   position       normal         uv
        constexpr auto default_vertex_size = sizeof(Vec3) + sizeof(Vec3) + sizeof(Vec2);
        constexpr auto tangent_data_size = sizeof(Vec3);
        constexpr auto joint_data_size = sizeof(std::uint32_t) + sizeof(Vec4);
        const auto vertex_size_in_bytes = default_vertex_size + (needs_tangents ? tangent_data_size : 0) + (needs_joint_data ? joint_data_size : 0);
        const auto vertex_buffer_size_in_bytes = model.vertices.size() * vertex_size_in_bytes;
        auto vertex_buffer = std::make_unique<std::uint8_t[]>(vertex_buffer_size_in_bytes);
        auto buffer_ptr = vertex_buffer.get();
        for (const auto& vertex : model.vertices)
        {
            std::memcpy(buffer_ptr, &vertex, default_vertex_size);
            buffer_ptr += default_vertex_size;
            if (needs_tangents)
            {
                std::memcpy(buffer_ptr, &vertex.tangent, tangent_data_size);
                buffer_ptr += tangent_data_size;
            }
            if (needs_joint_data)
            {
                std::memcpy(buffer_ptr, &vertex.joint_indices, joint_data_size);
                buffer_ptr += joint_data_size;
            }
        }

        file.write((const char*)vertex_buffer.get(), vertex_buffer_size_in_bytes);
    }

    file.write((const char*)model.indices.data(), sizeof(model.indices.front()) * model.indices.size());

    for (auto& material : model.materials)
    {
        file.write((const char*)&material.diffuse_coefficient, sizeof(material.diffuse_coefficient));
        file.write((const char*)&material.specular_coefficient, sizeof(material.specular_coefficient));
        file.write((const char*)&material.shininess, sizeof(material.shininess));
        file.write((const char*)&material.flags, sizeof(material.flags));
        // size + 1 for strings to include null terminator
        file.write(material.diffuse_map_filename.c_str(), material.diffuse_map_filename.size() + 1); 
        file.write(material.specular_map_filename.c_str(), material.specular_map_filename.size() + 1); 
        file.write(material.normal_map_filename.c_str(), material.normal_map_filename.size() + 1); 
    }
}

void PushChildrenOf(std::vector<AssimpNode>& nodes, unsigned int begin)
{
    unsigned int end = (unsigned int)nodes.size();
    for (unsigned int i = begin; i < end; i++)
    {
        auto node = nodes[i];
        for (unsigned int j = 0; j < node.node->mNumChildren; j++)
        {
            nodes.push_back({ node.node->mChildren[j], (int)i });
        }
    }
    if (end < nodes.size())
    {
        PushChildrenOf(nodes, end);
    }
}

std::vector<AssimpNode> FlattenNodeHierarchy(aiNode* root)
{
    std::vector<AssimpNode> nodes;
    nodes.push_back({ root, -1 });
    PushChildrenOf(nodes, 0);
    return nodes;
}

Mat3x4 ToGLMMat(const aiMatrix4x4& assimp_matrix)
{
    Mat3x4 glm_mat;

    glm_mat.column1[0] = assimp_matrix.a1;
    glm_mat.column1[1] = assimp_matrix.b1;
    glm_mat.column1[2] = assimp_matrix.c1;

    glm_mat.column2[0] = assimp_matrix.a2;
    glm_mat.column2[1] = assimp_matrix.b2;
    glm_mat.column2[2] = assimp_matrix.c2;

    glm_mat.column3[0] = assimp_matrix.a3;
    glm_mat.column3[1] = assimp_matrix.b3;
    glm_mat.column3[2] = assimp_matrix.c3;

    glm_mat.column4[0] = assimp_matrix.a4;
    glm_mat.column4[1] = assimp_matrix.b4;
    glm_mat.column4[2] = assimp_matrix.c4;

    return glm_mat;
}

SkeletonFile LoadSkeleton(const std::string_view path)
{
    std::ifstream file(path.data(), std::ios::binary);

    SkeletonFile data;
    file.read((char*)&data.header, sizeof(data.header));
    assert(data.header.magic_number == SkeletonFile::correct_magic_number);
    const auto num_joints = (int)data.header.num_joints;
    data.joints.resize(num_joints);
    file.read((char*)&data.joints[0], num_joints * sizeof(data.joints[0]));
    data.joint_names.resize(num_joints);
    for (auto& name : data.joint_names)
    {
        std::getline(file, name, '\0');
    }

    return data;
}
