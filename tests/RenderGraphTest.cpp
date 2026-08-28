#include "renderer/render_graph/RenderGraph.hpp"

#include <volk.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

renderer::render_graph::RenderGraph makeGraph() {
    return renderer::render_graph::RenderGraph({.graphics = 0, .compute = 0, .transfer = 0});
}

void testOrderingAndLifetimes() {
    namespace rg = renderer::render_graph;
    auto graph = makeGraph();
    constexpr VkImageSubresourceRange colorRange{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    const rg::ImageHandle image =
        graph.importImage("Color", VK_NULL_HANDLE, colorRange, {}, rg::QueueType::graphics);
    const rg::BufferHandle buffer =
        graph.importBuffer("Work", VK_NULL_HANDLE, 0, 256, {}, rg::QueueType::graphics);

    const rg::PassHandle produceImage = graph.addPass("Produce image", rg::QueueType::graphics);
    graph.writeImage(produceImage, image,
                     {
                         .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         .access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                         .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     });
    const rg::PassHandle inspectBuffer =
        graph.addPass("Inspect old buffer", rg::QueueType::compute);
    graph.readBuffer(inspectBuffer, buffer,
                     {
                         .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         .access = VK_ACCESS_2_SHADER_READ_BIT,
                     });
    const rg::PassHandle convert = graph.addPass("Convert", rg::QueueType::compute);
    graph.readImage(convert, image,
                    {
                        .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                        .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    });
    graph.writeBuffer(convert, buffer,
                      {
                          .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      });
    const rg::PassHandle consume = graph.addPass("Consume", rg::QueueType::graphics);
    graph.readBuffer(consume, buffer,
                         {
                             .stages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                             .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                         });
    graph.addDependency(produceImage, consume);

    graph.compile();
    const std::vector<std::string> expected{"Produce image", "Inspect old buffer", "Convert",
                                            "Consume"};
    require(std::vector<std::string>(graph.orderedPassNames().begin(),
                                     graph.orderedPassNames().end()) == expected,
            "Render-graph ordering is incorrect");
    const rg::ResourceLifetime imageLifetime = graph.lifetime(image);
    require(imageLifetime.used && imageLifetime.firstPass == 0 && imageLifetime.lastPass == 2,
            "Image lifetime analysis is incorrect");
    const rg::ResourceLifetime bufferLifetime = graph.lifetime(buffer);
    require(bufferLifetime.used && bufferLifetime.firstPass == 1 && bufferLifetime.lastPass == 3,
            "Buffer lifetime analysis is incorrect");
}

void testCycleDetection() {
    namespace rg = renderer::render_graph;
    auto graph = makeGraph();
    const rg::PassHandle first = graph.addPass("First", rg::QueueType::graphics);
    const rg::PassHandle second = graph.addPass("Second", rg::QueueType::graphics);
    graph.addDependency(first, second);
    graph.addDependency(second, first);
    bool rejected = false;
    try {
        graph.compile();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "Render-graph dependency cycle was not rejected");
}

void testDeclarationValidation() {
    namespace rg = renderer::render_graph;
    auto graph = makeGraph();
    const rg::BufferHandle buffer =
        graph.importBuffer("Buffer", VK_NULL_HANDLE, 0, 64, {}, rg::QueueType::graphics);
    const rg::PassHandle pass = graph.addPass("Pass", rg::QueueType::graphics);
    const rg::BufferState state{
        .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .access = VK_ACCESS_2_SHADER_READ_BIT,
    };
    graph.readBuffer(pass, buffer, state);
    bool rejected = false;
    try {
        graph.writeBuffer(pass, buffer, state);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "Duplicate pass resource declaration was not rejected");

    auto mismatchedGraph = makeGraph();
    const rg::BufferHandle mismatchedBuffer =
        mismatchedGraph.importBuffer("Buffer", VK_NULL_HANDLE, 0, 64, {}, rg::QueueType::graphics);
    const rg::PassHandle mismatchedPass = mismatchedGraph.addPass("Pass", rg::QueueType::graphics);
    rejected = false;
    try {
        mismatchedGraph.writeBuffer(mismatchedPass, mismatchedBuffer, state);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "Write declaration with read-only access flags was not rejected");
}

} // namespace

int main() {
    try {
        testOrderingAndLifetimes();
        testCycleDetection();
        testDeclarationValidation();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
