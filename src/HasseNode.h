#pragma once

#include <vector>
#include <unordered_map>

struct HasseNode {
  double weight;
  std::vector<int> word;
  std::vector<HasseNode*> faces;
	std::vector<HasseNode*> cofaces;

  HasseNode();

  ~HasseNode();

  HasseNode(const std::vector<int>& node);

  void UpdateWeight(double new_weight);
};