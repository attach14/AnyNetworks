#include "HyperGraph.h"
#include <algorithm>

void HyperGraph::AddEdge(std::vector<int> edge) {
  hasse_.InsertNode(edge);
}

void HyperGraph::RemoveEdge(std::vector<int> edge) {
  hasse_.RemoveNode(edge);
}

std::vector<std::vector<int>> HyperGraph::GetEdges() {
  return this->GetMaxElements();
}
