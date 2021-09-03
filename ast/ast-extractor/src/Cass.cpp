/*
MIT License

Copyright (c) 2021 Intel Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <iostream>
#include <unordered_map>

#include "Cass.h"

Cass::Cass(Type type, std::string label, CassVec &children)
    : type(type), label(label), children(std::move(children)) {}

Cass::Cass(Type type, std::string label) : type(type), label(label) {}

void Cass::dump() const { dumpRec(0); }

void Cass::dumpRec(int depth) const {
  for (int i = 0; i < depth; i++)
    std::cerr << "  ";
  std::cerr << "<" << this << "> ";
  std::cerr << label;
  if (type == Type::LocalVar)
    std::cerr << " #Local PrevUse: <" << prevUse << "> NextUse: <" << nextUse
              << ">";
  std::cerr << std::endl;
  for (const auto &child : children) {
    if (child == nullptr) {
      for (int i = 0; i < depth + 1; i++)
        std::cerr << "  ";
      std::cerr << "NULL" << std::endl;
    } else {
      child->dumpRec(depth + 1);
    }
  }
}

void gatherNodes(const Cass *cass, std::vector<const Cass *> &nodes) {
  nodes.push_back(cass);
  for (const auto &c : cass->children)
    gatherNodes(c.get(), nodes);
}

void Cass::serialize(const Cass *cass, std::ostream &os) {
  std::vector<const Cass *> nodes;
  std::unordered_map<const Cass *, int> node2id;

  gatherNodes(cass, nodes);

  int i = 0;
  for (auto n : nodes)
    node2id[n] = i++;
  if (node2id.find(nullptr) != node2id.end())
    throw std::runtime_error("Unexpected nullptr in node2id");
  node2id[nullptr] = -1;

  os << nodes.size() << '\t';
  for (auto n : nodes) {
    switch (n->type) {
    case Type::NumLit:
      os << 'N' << n->label << '\t';
      break;
    case Type::CharLit:
      os << 'C' << n->label << '\t';
      break;
    case Type::StringLit:
      os << 'S' << n->label << '\t';
      break;
    case Type::GlobalVar:
      os << 'V' << n->label << '\t';
      break;
    case Type::GlobalFun:
      os << 'F' << n->label << '\t';
      break;
    case Type::LocalVar:
      os << 'v' << n->label << '\t' << node2id[n->prevUse] << '\t'
         << node2id[n->nextUse] << '\t';
      break;
    case Type::LocalFun:
      os << 'f' << n->label << '\t' << node2id[n->prevUse] << '\t'
         << node2id[n->nextUse] << '\t';
      break;
    case Type::Internal:
      os << 'I' << n->label << '\t' << n->children.size() << '\t';
      break;
    case Type::Error:
      os << 'E' << '\t';
      break;
    default:
      throw std::runtime_error("Unexpected CASS node type");
      break;
    }
  }
}
