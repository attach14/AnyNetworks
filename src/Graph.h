#pragma once

#include "AbstractModel.h"

class Graph : public AbstractModel {
 public:
  void AddEdge(int v, int u);

  void RemoveEdge(int v, int u);

  std::vector<std::vector<int>> GetEdges();
};