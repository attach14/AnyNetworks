#pragma once

#include "AbstractModel.h"

class HyperGraph : public AbstractModel {
 public:
  void AddEdge(std::vector<int> edge);

  void RemoveEdge(std::vector<int> edge);

  std::vector<std::vector<int>> GetEdges();
};