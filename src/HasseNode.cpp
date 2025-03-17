#include "HasseNode.h"
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_set>

HasseNode::HasseNode() {
}

HasseNode::~HasseNode() {
  for (auto face : faces) {
      for (auto iter = face->cofaces.begin(); iter != face->cofaces.end();) {
          if (*iter == this) {
              iter = face->cofaces.erase(iter);
          } else {
              ++iter;
          }
      }
  }
  for (HasseNode* coface : cofaces) {
      delete coface;
  }
}

HasseNode::HasseNode(const std::vector<int>& node) {
  word = node;
  sort(word.begin(), word.end());
}


void HasseNode::UpdateWeight(double new_weight) {
  if (new_weight <= 0) {
    throw std::runtime_error("Weight must be positive");
  }
  weight = new_weight;
}

