#include "internal/gui.hpp"

#include <imgui.h>
#include <ImGuiFileDialog.h>
#include <fmt/core.h>

#include <array> // std::array

namespace mimir
{

static std::array<ViewType, 4> kAllViewTypes = {
    ViewType::Markers,
    ViewType::Edges,
    ViewType::Voxels,
    ViewType::Image
};
static std::array<DomainType, 2> kAllDomains = {
    DomainType::Structured,
    DomainType::Unstructured
};
/*
static std::array<ResourceType, 4> kAllResources = {
    ResourceType::Buffer,
    ResourceType::IndexBuffer,
    ResourceType::Texture,
    ResourceType::LinearTexture
};
static std::array<ComponentType, 7> kAllComponentTypes = {
    ComponentType::Int,
    ComponentType::Long,
    ComponentType::Short,
    ComponentType::Char,
    ComponentType::Float,
    ComponentType::Double,
    ComponentType::Half
};*/

struct AllResources
{
    static bool ItemGetter(void* data, int n, const char** out_str)
    {
        *out_str = getResourceType(((ResourceType*)data)[n]);
        return true;
    }
};
struct AllDomains
{
    static bool ItemGetter(void* data, int n, const char** out_str)
    {
        *out_str = getDomainType(((DomainType*)data)[n]);
        return true;
    }
};
struct AllViewTypes
{
    static bool ItemGetter(void* data, int n, const char** out_str)
    {
        *out_str = getViewType(((ViewType*)data)[n]);
        return true;
    }
};
struct AllComponentTypes
{
    static bool ItemGetter(void* data, int n, const char** out_str)
    {
        *out_str = getComponentType(((ComponentType*)data)[n]);
        return true;
    }
};

std::string getExtent(uint3 extent, DataDomain domain)
{
    switch (domain)
    {
        case DataDomain::Domain2D:
        {
            return fmt::format("({},{})", extent.x, extent.y);
        }
        case DataDomain::Domain3D:
        {
            return fmt::format("({},{},{})", extent.x, extent.y, extent.z);
        }
        default: return "unknown";
    }
}

void addTableRow(const std::string& key, const std::string& value)
{
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::AlignTextToFramePadding();
    ImGui::Text("%s", key.c_str());
    ImGui::TableSetColumnIndex(1);
    ImGui::Text("%s", value.c_str());
}

bool addTableRowCombo(const std::string& key, int* current_item,
    bool(*items_getter)(void* data, int idx, const char** out_text),
    void* data, int items_count)
{
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::AlignTextToFramePadding();
    ImGui::Text("%s", key.c_str());
    ImGui::TableSetColumnIndex(1);
    return ImGui::Combo(key.c_str(), current_item, items_getter, data, items_count);
}

void addViewObjectGui(InteropView *view_ptr, int uid)
{
    ImGui::PushID(view_ptr);
    auto& params = view_ptr->params;
    //bool node_open = ImGui::TreeNode("Object", "%s_%u", "View", uid);
    bool node_open = ImGui::CollapsingHeader("", ImGuiTreeNodeFlags_AllowItemOverlap);
    ImGui::SameLine(); ImGui::Text("%s #%u", "View", uid);
    ImGui::SameLine(ImGui::GetWindowWidth()-60); ImGui::Checkbox("show", &params.options.visible);
    if (node_open)
    {
        bool type_check = ImGui::Combo("View type", (int*)&params.view_type,
            &AllViewTypes::ItemGetter, kAllViewTypes.data(), kAllViewTypes.size()
        );
        if (type_check) printf("View %d: switched view type to %s\n", uid, getViewType(params.view_type));
        bool dom_check = ImGui::Combo("Domain type", (int*)&params.domain_type,
            &AllDomains::ItemGetter, kAllDomains.data(), kAllDomains.size()
        );
        if (dom_check) printf("View %d: switched domain type to %s\n", uid, getDomainType(params.domain_type));
        ImGui::SliderFloat("Element size (px)", &params.options.default_size, 1.f, 100.f);
        ImGui::ColorEdit4("Element color", (float*)&params.options.default_color);
        ImGui::SliderFloat("depth", &params.options.depth, 0.f, 1.f);

        int min_instance = 0;
        int max_instance = params.instance_count - 1;
        ImGui::SliderScalar("Instance index", ImGuiDataType_S32, &params.options.instance_index, &min_instance, &max_instance);


        if (ImGui::BeginTable("split", 2, ImGuiTableFlags_BordersOuter | ImGuiTableFlags_Resizable))
        {
            addTableRow("Data domain", getDataDomain(params.data_domain));
            addTableRow("Data extent", getExtent(params.extent, params.data_domain));
            ImGui::EndTable();
        }
        for (const auto &[attr, memory] : params.attributes)
        {
            if (ImGui::BeginTable("split", 2, ImGuiTableFlags_BordersOuter | ImGuiTableFlags_Resizable))
            {
                auto& info = memory.params;
                //addTableRow("Element count", std::to_string(info.element_count));
                addTableRow("Attribute type", getAttributeType(attr));
                addTableRow("Resource type", getResourceType(info.resource_type));
                addTableRow("Data type", getComponentType(info.component_type));
                addTableRow("Channel count", std::to_string(info.channel_count));
                addTableRow("Data layout", getDataLayout(info.layout));

                /*bool res_check = addTableRowCombo("Resource type", (int*)&info.resource_type,
                    &AllResources::ItemGetter, kAllResources.data(), kAllResources.size()
                );
                if (res_check) printf("View %d: switched resource type to %s\n", uid, getResourceType(info.resource_type));
                bool data_check = addTableRowCombo("Data type", (int*)&info.component_type,
                    &AllComponentTypes::ItemGetter, kAllComponentTypes.data(), kAllComponentTypes.size()
                );
                if (data_check) printf("View %d: switched data type to %s\n", uid, getComponentType(info.component_type));*/

                ImGui::EndTable();
            }
        }
        //ImGui::TreePop();
    }
    ImGui::PopID();
}

} // namespace mimir