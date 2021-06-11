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
