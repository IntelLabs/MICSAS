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
#include <fstream>
#include <sstream>

#include "CassExtractorBase.h"

CassExtractorBase::CassExtractorBase(bool allowParseErrors) {
  this->allowParseErrors = allowParseErrors;
  parser = ts_parser_new();
}

CassExtractorBase::~CassExtractorBase() { ts_parser_delete(parser); }

CassVec CassExtractorBase::extractCassForFunctions(const std::string &src) {
  srcBytes = src;

  TSTree *tree = ts_parser_parse_string(parser, nullptr, srcBytes.c_str(),
                                        srcBytes.length());

  auto root_node = ts_tree_root_node(tree);

  if (!allowParseErrors) {
    if (ts_node_has_error(root_node)) {
      CassVec ret = {};
      ret.push_back(nullptr);
      return ret;
    }
  }

  functions.clear();
  collectFunctions(root_node);

  CassVec casses;
  for (const auto &function : functions) {
    symbolTable.clear();
    node2symbol.clear();
    identifyLocalVariables(function);

    node2cass.clear();
    auto cass = buildCass(function);
    if (cass) {
      casses.push_back(std::move(cass));
    }
  }

  ts_tree_delete(tree);

  return casses;
}
