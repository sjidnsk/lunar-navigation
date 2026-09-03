#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "lunar_incremental_navigation_core/elevation_map.hpp"

namespace lunar::incremental_navigation {

template <typename Tile>
class PersistentTileDirectory final {
 public:
  using TilePtr = std::shared_ptr<const Tile>;

  explicit PersistentTileDirectory(SparseGridGeometry geometry)
      : geometry_(std::move(geometry)) {
    if (!geometry_.valid()) {
      throw std::invalid_argument(
          "persistent tile directory requires valid sparse geometry");
    }
  }

  [[nodiscard]] const SparseGridGeometry& geometry() const noexcept {
    return geometry_;
  }

  [[nodiscard]] TilePtr Find(const TileIndex index) const noexcept {
    NodePtr node = root_;
    while (node) {
      if (index < node->index) {
        node = node->left;
      } else if (node->index < index) {
        node = node->right;
      } else {
        return node->tile;
      }
    }
    return nullptr;
  }

  [[nodiscard]] std::size_t tile_count() const noexcept {
    return tile_count_;
  }

  [[nodiscard]] std::size_t last_update_copied_nodes() const noexcept {
    return last_update_copied_nodes_;
  }

  [[nodiscard]] std::vector<TileIndex> tile_indices() const {
    std::vector<TileIndex> indices;
    indices.reserve(tile_count_);
    CollectIndices(root_, indices);
    return indices;
  }

  [[nodiscard]] PersistentTileDirectory WithTile(const TileIndex index,
                                                 TilePtr tile) const {
    if (!tile || !geometry_.Contains(index)) {
      throw std::invalid_argument(
          "persistent tile directory rejects null or out-of-bounds tiles");
    }
    ValidateHiddenBoundaryCells(index, *tile);
    std::size_t copied_nodes = 0U;
    bool inserted = false;
    NodePtr root = Upsert(root_, index, tile, copied_nodes, inserted);
    return PersistentTileDirectory(
        geometry_, std::move(root),
        tile_count_ + static_cast<std::size_t>(inserted), copied_nodes);
  }

  [[nodiscard]] PersistentTileDirectory WithGeometry(
      SparseGridGeometry geometry) const {
    if (!geometry.valid() || !IsLatticeAlignedSuperset(geometry)) {
      throw std::invalid_argument(
          "persistent tile directory geometry must expand one lattice");
    }
    return PersistentTileDirectory(std::move(geometry), root_, tile_count_,
                                   0U);
  }

  [[nodiscard]] bool shares_root_with(
      const PersistentTileDirectory& other) const noexcept {
    return root_ == other.root_;
  }

 private:
  struct Node;
  using NodePtr = std::shared_ptr<const Node>;

  struct Node final {
    Node(const TileIndex index_value, TilePtr tile_value, NodePtr left_value,
         NodePtr right_value)
        : index(index_value),
          tile(std::move(tile_value)),
          left(std::move(left_value)),
          right(std::move(right_value)),
          height(1 + std::max(left ? left->height : 0,
                              right ? right->height : 0)) {}

    TileIndex index;
    TilePtr tile;
    NodePtr left;
    NodePtr right;
    int height{};
  };

  PersistentTileDirectory(SparseGridGeometry geometry, NodePtr root,
                          const std::size_t tile_count,
                          const std::size_t copied_nodes) noexcept
      : geometry_(std::move(geometry)),
        root_(std::move(root)),
        tile_count_(tile_count),
        last_update_copied_nodes_(copied_nodes) {}

  [[nodiscard]] bool IsLatticeAlignedSuperset(
      const SparseGridGeometry& candidate) const noexcept {
    const Vec3 origin = geometry_.origin_m();
    const Vec3 candidate_origin = candidate.origin_m();
    const GridIndex min = geometry_.min_inclusive();
    const GridIndex max = geometry_.max_exclusive();
    const GridIndex candidate_min = candidate.min_inclusive();
    const GridIndex candidate_max = candidate.max_exclusive();
    return candidate.frame_id() == geometry_.frame_id() &&
           candidate.resolution_m() == geometry_.resolution_m() &&
           candidate_origin == origin && candidate_min.x <= min.x &&
           candidate_min.y <= min.y && candidate_max.x >= max.x &&
           candidate_max.y >= max.y;
  }

  void ValidateHiddenBoundaryCells(const TileIndex index,
                                   const Tile& tile) const {
    const std::int64_t tile_min_x = index.x * kGridTileWidthCells;
    const std::int64_t tile_min_y = index.y * kGridTileWidthCells;
    const std::int64_t tile_max_x =
        tile_min_x + (kGridTileWidthCells - 1);
    const std::int64_t tile_max_y =
        tile_min_y + (kGridTileWidthCells - 1);
    const GridIndex min = geometry_.min_inclusive();
    const GridIndex max = geometry_.max_exclusive();
    if (tile_min_x >= min.x && tile_min_y >= min.y && tile_max_x < max.x &&
        tile_max_y < max.y) {
      return;
    }
    for (std::size_t offset = 0U; offset < kGridTileCellCount; ++offset) {
      const std::int64_t cell_x =
          tile_min_x + static_cast<std::int64_t>(
                           offset % static_cast<std::size_t>(
                                        kGridTileWidthCells));
      const std::int64_t cell_y =
          tile_min_y + static_cast<std::int64_t>(
                           offset / static_cast<std::size_t>(
                                        kGridTileWidthCells));
      const bool inside = cell_x >= min.x && cell_x < max.x &&
                          cell_y >= min.y && cell_y < max.y;
      if (!inside && !tile.IsUnobserved(offset)) {
        throw std::invalid_argument(
            "persistent boundary tile hides observed cells outside bounds");
      }
    }
  }

  [[nodiscard]] static int Height(const NodePtr& node) noexcept {
    return node ? node->height : 0;
  }

  [[nodiscard]] static int BalanceFactor(const NodePtr& node) noexcept {
    return node ? Height(node->left) - Height(node->right) : 0;
  }

  [[nodiscard]] static NodePtr MakeNode(
      const TileIndex index, const TilePtr& tile, NodePtr left, NodePtr right,
      std::size_t& copied_nodes) {
    ++copied_nodes;
    return std::make_shared<const Node>(index, tile, std::move(left),
                                        std::move(right));
  }

  [[nodiscard]] static NodePtr RotateLeft(
      const NodePtr& root, std::size_t& copied_nodes) {
    const NodePtr pivot = root->right;
    const NodePtr new_left =
        MakeNode(root->index, root->tile, root->left, pivot->left,
                 copied_nodes);
    return MakeNode(pivot->index, pivot->tile, new_left, pivot->right,
                    copied_nodes);
  }

  [[nodiscard]] static NodePtr RotateRight(
      const NodePtr& root, std::size_t& copied_nodes) {
    const NodePtr pivot = root->left;
    const NodePtr new_right =
        MakeNode(root->index, root->tile, pivot->right, root->right,
                 copied_nodes);
    return MakeNode(pivot->index, pivot->tile, pivot->left, new_right,
                    copied_nodes);
  }

  [[nodiscard]] static NodePtr Balance(NodePtr root,
                                       std::size_t& copied_nodes) {
    const int balance = BalanceFactor(root);
    if (balance > 1) {
      if (BalanceFactor(root->left) < 0) {
        const NodePtr rotated_left = RotateLeft(root->left, copied_nodes);
        root = MakeNode(root->index, root->tile, rotated_left, root->right,
                        copied_nodes);
      }
      return RotateRight(root, copied_nodes);
    }
    if (balance < -1) {
      if (BalanceFactor(root->right) > 0) {
        const NodePtr rotated_right = RotateRight(root->right, copied_nodes);
        root = MakeNode(root->index, root->tile, root->left, rotated_right,
                        copied_nodes);
      }
      return RotateLeft(root, copied_nodes);
    }
    return root;
  }

  [[nodiscard]] static NodePtr Upsert(
      const NodePtr& root, const TileIndex index, const TilePtr& tile,
      std::size_t& copied_nodes, bool& inserted) {
    if (!root) {
      inserted = true;
      return MakeNode(index, tile, nullptr, nullptr, copied_nodes);
    }
    if (index < root->index) {
      const NodePtr left =
          Upsert(root->left, index, tile, copied_nodes, inserted);
      if (left == root->left) {
        return root;
      }
      return Balance(MakeNode(root->index, root->tile, left, root->right,
                              copied_nodes),
                     copied_nodes);
    }
    if (root->index < index) {
      const NodePtr right =
          Upsert(root->right, index, tile, copied_nodes, inserted);
      if (right == root->right) {
        return root;
      }
      return Balance(MakeNode(root->index, root->tile, root->left, right,
                              copied_nodes),
                     copied_nodes);
    }
    return root->tile == tile
               ? root
               : MakeNode(index, tile, root->left, root->right, copied_nodes);
  }

  static void CollectIndices(const NodePtr& root,
                             std::vector<TileIndex>& indices) {
    if (!root) {
      return;
    }
    CollectIndices(root->left, indices);
    indices.push_back(root->index);
    CollectIndices(root->right, indices);
  }

  SparseGridGeometry geometry_;
  NodePtr root_;
  std::size_t tile_count_{};
  std::size_t last_update_copied_nodes_{};
};

}  // namespace lunar::incremental_navigation
